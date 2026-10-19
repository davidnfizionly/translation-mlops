# Endpoint stack: the real-time endpoint alone, created for one demo session
# and destroyed right after. It bills per instance hour while it exists.

terraform {
  required_version = ">= 1.10"

  backend "s3" {
    key = "endpoint/terraform.tfstate"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.69"
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
  description = "Terraform state bucket (from infra/backend.hcl), used to read the ml stack outputs."
}

variable "model_package_version" {
  type        = number
  default     = 1
  description = "Approved version of the model package group to deploy."
}

variable "instance_type" {
  type    = string
  default = "ml.g4dn.xlarge"
}

data "terraform_remote_state" "ml" {
  backend = "s3"
  config = {
    bucket = var.state_bucket
    key    = "ml/terraform.tfstate"
    region = var.region
  }
}

data "aws_partition" "current" {}
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name          = "translation-mlops"
  endpoint_name = "${local.name}-nllb-en-din-nus"
  partition     = data.aws_partition.current.partition
  account       = data.aws_caller_identity.current.account_id
  region        = data.aws_region.current.region
  ml_bucket_arn = "arn:${local.partition}:s3:::${data.terraform_remote_state.ml.outputs.ml_bucket}"
  model_package = "arn:${local.partition}:sagemaker:${local.region}:${local.account}:model-package/${data.terraform_remote_state.ml.outputs.model_package_group}/${var.model_package_version}"
  # AWS Deep Learning Containers registry account for us-east-1 (Hugging Face inference image).
  inference_repo = "arn:${local.partition}:ecr:${local.region}:763104351884:repository/huggingface-pytorch-inference"
}

# ---------------------------------------------------------------- Endpoint role

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "endpoint" {
  statement {
    sid       = "ReadModelArtifact"
    actions   = ["s3:GetObject"]
    resources = ["${local.ml_bucket_arn}/training/*"]
  }

  statement {
    sid       = "WriteDataCapture"
    actions   = ["s3:PutObject"]
    resources = ["${local.ml_bucket_arn}/datacapture/*"]
  }

  statement {
    sid       = "PullImageToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "PullInferenceImage"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [local.inference_repo]
  }

  statement {
    sid       = "EndpointLogs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:${local.partition}:logs:${local.region}:${local.account}:log-group:/aws/sagemaker/Endpoints/${local.endpoint_name}*"]
  }

  statement {
    sid       = "EndpointMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }
}

resource "aws_iam_role" "endpoint" {
  name               = "${local.name}-endpoint"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "endpoint" {
  name   = "endpoint"
  role   = aws_iam_role.endpoint.id
  policy = data.aws_iam_policy_document.endpoint.json
}

# ---------------------------------------------------------------- Model, config, endpoint

# The model is built from the approved registry version, so the image and the
# model artifact are the ones the chrF gate evaluated.
resource "aws_sagemaker_model" "translation" {
  name               = "${local.name}-nllb-v${var.model_package_version}"
  execution_role_arn = aws_iam_role.endpoint.arn

  primary_container {
    model_package_name = local.model_package
  }

  depends_on = [aws_iam_role_policy.endpoint]
}

resource "aws_sagemaker_endpoint_configuration" "translation" {
  name = "${local.name}-nllb-v${var.model_package_version}"

  production_variants {
    variant_name                                      = "AllTraffic"
    model_name                                        = aws_sagemaker_model.translation.name
    instance_type                                     = var.instance_type
    initial_instance_count                            = 1
    model_data_download_timeout_in_seconds            = 1200
    container_startup_health_check_timeout_in_seconds = 900
  }

  # Request and response bodies for the drift report; stored with the bucket's SSE-S3 encryption.
  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${data.terraform_remote_state.ml.outputs.ml_bucket}/datacapture"

    capture_options {
      capture_mode = "Input"
    }

    capture_options {
      capture_mode = "Output"
    }

    capture_content_type_header {
      json_content_types = ["application/json"]
    }
  }
}

resource "aws_sagemaker_endpoint" "translation" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.translation.name
}

# ---------------------------------------------------------------- Outputs

output "endpoint_name" {
  value = aws_sagemaker_endpoint.translation.name
}

output "instance_type" {
  value = var.instance_type
}
