# Monitoring stack: one CloudWatch dashboard and two alarms for the endpoint.
# Alarms have no actions (no notifications); their state is shown on the dashboard.
# Custom metrics in the TranslationMlops namespace are published by the local proxy (console/server).

terraform {
  required_version = ">= 1.10"

  backend "s3" {
    key = "monitoring/terraform.tfstate"
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

variable "endpoint_name" {
  type    = string
  default = "translation-mlops-nllb-en-din-nus"
}

variable "latency_alarm_ms" {
  type        = number
  default     = 10000
  description = "p90 model latency above this value for 5 minutes sets the alarm."
}

locals {
  name    = "translation-mlops"
  variant = ["EndpointName", var.endpoint_name, "VariantName", "AllTraffic"]
  custom  = "TranslationMlops"
  region  = var.region
}

resource "aws_cloudwatch_metric_alarm" "errors_5xx" {
  alarm_name          = "${local.name}-endpoint-5xx"
  alarm_description   = "Any 5XX error returned by the translation endpoint in one minute."
  namespace           = "AWS/SageMaker"
  metric_name         = "Invocation5XXErrors"
  dimensions          = { EndpointName = var.endpoint_name, VariantName = "AllTraffic" }
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "latency_p90" {
  alarm_name          = "${local.name}-endpoint-latency-p90"
  alarm_description   = "p90 model latency above ${var.latency_alarm_ms} ms for 5 minutes."
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  dimensions          = { EndpointName = var.endpoint_name, VariantName = "AllTraffic" }
  extended_statistic  = "p90"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.latency_alarm_ms * 1000 # ModelLatency is in microseconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_dashboard" "endpoint" {
  dashboard_name = "${local.name}-endpoint"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 8, height = 6
        properties = {
          title = "Invocations and errors (per minute)", region = local.region, view = "timeSeries", stat = "Sum", period = 60
          metrics = [
            concat(["AWS/SageMaker", "Invocations"], local.variant),
            concat(["AWS/SageMaker", "Invocation4XXErrors"], local.variant),
            concat(["AWS/SageMaker", "Invocation5XXErrors"], local.variant),
          ]
        }
      },
      {
        type = "metric", x = 8, y = 0, width = 8, height = 6
        properties = {
          title = "Model latency (ms)", region = local.region, view = "timeSeries", period = 60
          metrics = [
            [{ expression = "m1 / 1000", label = "p50 ms", id = "e1" }],
            [{ expression = "m2 / 1000", label = "p90 ms", id = "e2" }],
            concat(["AWS/SageMaker", "ModelLatency"], local.variant, [{ stat = "p50", id = "m1", visible = false }]),
            concat(["AWS/SageMaker", "ModelLatency"], local.variant, [{ stat = "p90", id = "m2", visible = false }]),
          ]
        }
      },
      {
        type = "alarm", x = 16, y = 0, width = 8, height = 6
        properties = {
          title  = "Alarms (no notifications)"
          alarms = [aws_cloudwatch_metric_alarm.errors_5xx.arn, aws_cloudwatch_metric_alarm.latency_p90.arn]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 8, height = 6
        properties = {
          title = "Instance: GPU, GPU memory, CPU (%)", region = local.region, view = "timeSeries", stat = "Average", period = 60
          metrics = [
            concat(["/aws/sagemaker/Endpoints", "GPUUtilization"], local.variant),
            concat(["/aws/sagemaker/Endpoints", "GPUMemoryUtilization"], local.variant),
            concat(["/aws/sagemaker/Endpoints", "CPUUtilization"], local.variant),
          ]
        }
      },
      {
        type = "metric", x = 8, y = 6, width = 8, height = 6
        properties = {
          title = "Drift signals: input words and output/input word ratio", region = local.region, view = "timeSeries", stat = "Average", period = 60
          metrics = [
            [local.custom, "InputWords", "EndpointName", var.endpoint_name],
            [local.custom, "OutputInputRatio", "EndpointName", var.endpoint_name, { yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 16, y = 6, width = 8, height = 6
        properties = {
          title = "Estimated cost attributed to requests (USD, estimate)", region = local.region, view = "timeSeries", stat = "Sum", period = 300
          metrics = [
            [local.custom, "EstimatedRequestCostUSD", "EndpointName", var.endpoint_name],
          ]
        }
      },
    ]
  })
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.endpoint.dashboard_name
}
