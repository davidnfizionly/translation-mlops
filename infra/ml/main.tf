# ML stack: artifact bucket, one IAM role per stage (training, evaluation),
# and the Model Registry group. Nothing here bills while idle except S3 storage.

terraform {
  required_version = ">= 1.10"

  backend "s3" {
    key = "ml/terraform.tfstate"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.69"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project = "translation-mlops"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "state_bucket" {
  type        = string
  description = "Terraform state bucket (from infra/backend.hcl), used to read the data stack outputs."
}

data "terraform_remote_state" "data" {
  backend = "s3"
  config = {
    bucket = var.state_bucket
    key    = "data/terraform.tfstate"
    region = var.region
  }
}

data "aws_partition" "current" {}
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name        = "translation-mlops"
  data_bucket = data.terraform_remote_state.data.outputs.data_bucket
  data_arn    = "arn:${data.aws_partition.current.partition}:s3:::${local.data_bucket}"
  account     = data.aws_caller_identity.current.account_id
  region      = data.aws_region.current.region
  # AWS Deep Learning Containers registry account for us-east-1 (Hugging Face images).
  dlc_repos = [
    "arn:${data.aws_partition.current.partition}:ecr:${local.region}:763104351884:repository/huggingface-pytorch-training",
    "arn:${data.aws_partition.current.partition}:ecr:${local.region}:763104351884:repository/huggingface-pytorch-inference",
  ]
}

# ---------------------------------------------------------------- Artifact bucket

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "ml" {
  bucket = "${local.name}-ml-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_ownership_controls" "ml" {
  bucket = aws_s3_bucket.ml.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "ml" {
  bucket = aws_s3_bucket.ml.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml" {
  bucket = aws_s3_bucket.ml.id

  rule {
    id     = "expire-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    expiration {
      days = 7
    }
  }

  rule {
    id     = "abort-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "ml_tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.ml.arn, "${aws_s3_bucket.ml.arn}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "ml" {
  bucket = aws_s3_bucket.ml.id
  policy = data.aws_iam_policy_document.ml_tls_only.json

  depends_on = [aws_s3_bucket_public_access_block.ml]
}

# ---------------------------------------------------------------- Shared policy pieces

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account]
    }
  }
}

# ---------------------------------------------------------------- Training role

data "aws_iam_policy_document" "training" {
  statement {
    sid       = "ReadTrainValidSplits"
    actions   = ["s3:GetObject"]
    resources = ["${local.data_arn}/processed/split=train/*", "${local.data_arn}/processed/split=valid/*"]
  }

  statement {
    sid       = "ListDataSplits"
    actions   = ["s3:ListBucket"]
    resources = [local.data_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["processed/split=train/*", "processed/split=valid/*"]
    }
  }

  statement {
    sid       = "ReadCodeWriteArtifacts"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.ml.arn}/code/*", "${aws_s3_bucket.ml.arn}/training/*", "${aws_s3_bucket.ml.arn}/checkpoints/*"]
  }

  statement {
    sid       = "ListArtifactPrefixes"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.ml.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["code/*", "training/*", "checkpoints/*"]
    }
  }

  statement {
    sid       = "TrainingLogs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:${data.aws_partition.current.partition}:logs:${local.region}:${local.account}:log-group:/aws/sagemaker/TrainingJobs*"]
  }

  statement {
    sid       = "TrainingMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "PullHuggingFaceImages"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = local.dlc_repos
  }
}

resource "aws_iam_role" "training" {
  name               = "${local.name}-training"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "training" {
  name   = "training"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}

# ---------------------------------------------------------------- Evaluation role (processing jobs)

data "aws_iam_policy_document" "evaluation" {
  statement {
    sid       = "ReadTestSplit"
    actions   = ["s3:GetObject"]
    resources = ["${local.data_arn}/processed/split=test/*"]
  }

  statement {
    sid       = "ListTestSplit"
    actions   = ["s3:ListBucket"]
    resources = [local.data_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["processed/split=test/*"]
    }
  }

  statement {
    sid       = "ReadModelsAndCode"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.ml.arn}/code/*", "${aws_s3_bucket.ml.arn}/training/*", "${aws_s3_bucket.ml.arn}/evaluation/*"]
  }

  statement {
    sid       = "WriteEvaluation"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.ml.arn}/evaluation/*"]
  }

  statement {
    sid       = "ListArtifactPrefixes"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.ml.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["code/*", "training/*", "evaluation/*"]
    }
  }

  statement {
    sid       = "ProcessingLogs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:${data.aws_partition.current.partition}:logs:${local.region}:${local.account}:log-group:/aws/sagemaker/ProcessingJobs*"]
  }

  statement {
    sid       = "ProcessingMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "PullHuggingFaceImages"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = local.dlc_repos
  }
}

resource "aws_iam_role" "evaluation" {
  name               = "${local.name}-evaluation"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "evaluation" {
  name   = "evaluation"
  role   = aws_iam_role.evaluation.id
  policy = data.aws_iam_policy_document.evaluation.json
}

# ---------------------------------------------------------------- Model Registry

resource "aws_sagemaker_model_package_group" "translation" {
  model_package_group_name        = "${local.name}-nllb-en-din-nus"
  model_package_group_description = "NLLB-200 600M fine-tuned for English to Dinka and English to Nuer (dataset by Tajnaam, CC BY-NC 4.0; base model CC BY-NC 4.0)"
}

# ---------------------------------------------------------------- Outputs

output "ml_bucket" {
  value = aws_s3_bucket.ml.id
}

output "data_bucket" {
  value = local.data_bucket
}

output "training_role_name" {
  value = aws_iam_role.training.name
}

output "evaluation_role_name" {
  value = aws_iam_role.evaluation.name
}

output "model_package_group" {
  value = aws_sagemaker_model_package_group.translation.model_package_group_name
}
