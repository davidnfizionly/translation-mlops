# Data stack: data bucket, Glue Python shell job (clean, split, leakage check),
# Glue Data Catalog table over the splits, Athena workgroup and quality queries.

terraform {
  required_version = ">= 1.10"

  backend "s3" {
    key = "data/terraform.tfstate"
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

variable "split_seed" {
  type    = number
  default = 13
}

variable "test_size" {
  type    = number
  default = 2000
}

variable "valid_size" {
  type    = number
  default = 2000
}

locals {
  name          = "translation-mlops"
  raw_key       = "raw/nuer_dinka_translations_paired.json"
  wheel_name    = "translation_data_prep-0.1.0-py3-none-any.whl"
  wheel_path    = "${path.module}/../../build/${local.wheel_name}"
  job_script    = "${path.module}/../../data_prep/glue_job.py"
  quality_query = fileset("${path.module}/queries", "*.sql")
}

data "aws_partition" "current" {}
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ---------------------------------------------------------------- S3

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "data" {
  bucket = "${local.name}-data-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-athena-results"
    status = "Enabled"

    filter {
      prefix = "athena-results/"
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

data "aws_iam_policy_document" "data_tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]

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

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.data_tls_only.json

  depends_on = [aws_s3_bucket_public_access_block.data]
}

# Code only (the dataset itself is uploaded with the CLI, never through Terraform).
resource "aws_s3_object" "job_script" {
  bucket = aws_s3_bucket.data.id
  key    = "scripts/glue_job.py"
  source = local.job_script
  etag   = filemd5(local.job_script)
}

resource "aws_s3_object" "wheel" {
  bucket = aws_s3_bucket.data.id
  key    = "scripts/${local.wheel_name}"
  source = local.wheel_path
  etag   = filemd5(local.wheel_path)
}

# ---------------------------------------------------------------- IAM (Glue job role)

data "aws_iam_policy_document" "glue_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["glue.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "glue_data_prep" {
  name               = "${local.name}-glue-data-prep"
  assume_role_policy = data.aws_iam_policy_document.glue_assume.json
}

data "aws_iam_policy_document" "glue_data_prep" {
  statement {
    sid       = "ReadRawAndCode"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/raw/*", "${aws_s3_bucket.data.arn}/scripts/*"]
  }

  statement {
    sid       = "WriteSplitsAndReports"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/processed/*", "${aws_s3_bucket.data.arn}/reports/*"]
  }

  statement {
    sid       = "ListOwnPrefixes"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["raw/*", "scripts/*", "processed/*", "reports/*"]
    }
  }

  statement {
    sid     = "JobLogs"
    actions = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = [
      "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/python-jobs/*",
    ]
  }
}

resource "aws_iam_role_policy" "glue_data_prep" {
  name   = "data-prep"
  role   = aws_iam_role.glue_data_prep.id
  policy = data.aws_iam_policy_document.glue_data_prep.json
}

# ---------------------------------------------------------------- Glue job

resource "aws_glue_job" "data_prep" {
  name         = "${local.name}-data-prep"
  role_arn     = aws_iam_role.glue_data_prep.arn
  max_capacity = 1
  max_retries  = 0
  timeout      = 90

  command {
    name            = "pythonshell"
    python_version  = "3.9"
    script_location = "s3://${aws_s3_bucket.data.id}/${aws_s3_object.job_script.key}"
  }

  default_arguments = {
    "library-set"      = "analytics"
    "--extra-py-files" = "s3://${aws_s3_bucket.data.id}/${aws_s3_object.wheel.key}"
    "--bucket"         = aws_s3_bucket.data.id
    "--raw_key"        = local.raw_key
    "--seed"           = tostring(var.split_seed)
    "--test_size"      = tostring(var.test_size)
    "--valid_size"     = tostring(var.valid_size)
  }

  execution_property {
    max_concurrent_runs = 1
  }

  depends_on = [aws_iam_role_policy.glue_data_prep]
}

# ---------------------------------------------------------------- Catalog and Athena

resource "aws_glue_catalog_database" "corpus" {
  name        = "translation_mlops"
  description = "English-Nuer-Dinka corpus splits (dataset by Tajnaam, CC BY-NC 4.0)"
}

resource "aws_glue_catalog_table" "pairs" {
  name          = "pairs"
  database_name = aws_glue_catalog_database.corpus.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classification"            = "json"
    "projection.enabled"        = "true"
    "projection.split.type"     = "enum"
    "projection.split.values"   = "train,valid,test"
    "storage.location.template" = "s3://${aws_s3_bucket.data.id}/processed/split=$${split}/"
  }

  partition_keys {
    name = "split"
    type = "string"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data.id}/processed/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
    }

    columns {
      name = "row_id"
      type = "bigint"
    }
    columns {
      name = "english"
      type = "string"
    }
    columns {
      name = "nuer"
      type = "string"
    }
    columns {
      name = "din"
      type = "string"
    }
  }
}

resource "aws_athena_workgroup" "quality" {
  name          = local.name
  force_destroy = true

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = false
    bytes_scanned_cutoff_per_query     = 1073741824

    result_configuration {
      output_location = "s3://${aws_s3_bucket.data.id}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

resource "aws_athena_named_query" "quality" {
  for_each = local.quality_query

  name      = trimsuffix(each.value, ".sql")
  workgroup = aws_athena_workgroup.quality.id
  database  = aws_glue_catalog_database.corpus.name
  query     = file("${path.module}/queries/${each.value}")
}

# ---------------------------------------------------------------- Outputs

output "data_bucket" {
  value = aws_s3_bucket.data.id
}

output "glue_job" {
  value = aws_glue_job.data_prep.name
}

output "athena_workgroup" {
  value = aws_athena_workgroup.quality.name
}
