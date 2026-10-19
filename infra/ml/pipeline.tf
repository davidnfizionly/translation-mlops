# SageMaker Pipeline: Train -> Evaluate -> ChrfGate -> Register (or GateNotPassed).
# The definition is written here as JSON (same structure the SageMaker SDK
# emits), so no account-specific definition file is stored in the repository.
# A pipeline definition does not bill while idle; only executions do.

locals {
  training_image  = "763104351884.dkr.ecr.${local.region}.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
  inference_image = "763104351884.dkr.ecr.${local.region}.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.51.3-gpu-py312-cu124-ubuntu22.04"
  ml_s3           = "s3://${aws_s3_bucket.ml.id}"
  execution_id    = { Get = "Execution.PipelineExecutionId" }
  # Baseline report from the base-model evaluation job (proof/10-baseline-evaluation.txt).
  baseline_metrics = "${local.ml_s3}/evaluation/tm-eval-base-20261019-095652/metrics.json"
  eval_code_files  = ["run_eval.py", "gate.py", "requirements.txt", "pipeline_eval.sh"]
  sourcedir_path   = "${path.module}/../../build/train-sourcedir.tar.gz"
  pipeline_tags = [
    { Key = "Project", Value = "translation-mlops" },
    { Key = "Run", Value = "pipeline" },
  ]

  pipeline_definition = {
    Version  = "2020-12-01"
    Metadata = {}
    Parameters = [
      { Name = "TrainRows", Type = "String", DefaultValue = "50000" },
      { Name = "MaxSteps", Type = "String", DefaultValue = "3000" },
      { Name = "BaselineMetricsUri", Type = "String", DefaultValue = local.baseline_metrics },
    ]
    Steps = [
      {
        Name = "Train"
        Type = "Training"
        Arguments = {
          RoleArn = aws_iam_role.training.arn
          AlgorithmSpecification = {
            TrainingImage     = local.training_image
            TrainingInputMode = "File"
            MetricDefinitions = [
              { Name = "train:loss", Regex = "'loss': ([0-9.]+)" },
              { Name = "validation:loss", Regex = "'eval_loss': ([0-9.]+)" },
              { Name = "train:samples_per_second", Regex = "'train_samples_per_second': ([0-9.]+)" },
            ]
          }
          HyperParameters = {
            train_rows                  = { Get = "Parameters.TrainRows" }
            max_steps                   = { Get = "Parameters.MaxSteps" }
            valid_rows                  = "500"
            eval_steps                  = "500"
            save_steps                  = "500"
            warmup_steps                = "200"
            per_device_batch_size       = "8"
            gradient_accumulation_steps = "4"
            sagemaker_program           = "\"train.py\""
            sagemaker_submit_directory  = "\"${local.ml_s3}/${aws_s3_object.train_sourcedir.key}\""
            sagemaker_region            = "\"${local.region}\""
          }
          InputDataConfig = [for split in ["train", "valid"] : {
            ChannelName = split
            ContentType = "application/x-ndjson"
            DataSource = {
              S3DataSource = {
                S3DataType             = "S3Prefix"
                S3Uri                  = "s3://${local.data_bucket}/processed/split=${split}/"
                S3DataDistributionType = "FullyReplicated"
              }
            }
          }]
          OutputDataConfig  = { S3OutputPath = "${local.ml_s3}/training/pipeline" }
          CheckpointConfig  = { S3Uri = { "Std:Join" = { On = "/", Values = ["${local.ml_s3}/checkpoints/pipeline", local.execution_id] } } }
          ResourceConfig    = { InstanceType = "ml.g4dn.xlarge", InstanceCount = 1, VolumeSizeInGB = 50 }
          StoppingCondition = { MaxRuntimeInSeconds = 7200 }
          # Batch 8 x 4 and expandable segments: run 1 ran out of GPU memory at step 1,700 with batch 16 x 2.
          Environment = { PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
          Tags        = local.pipeline_tags
        }
      },
      {
        Name = "Evaluate"
        Type = "Processing"
        Arguments = {
          RoleArn = aws_iam_role.evaluation.arn
          AppSpecification = {
            ImageUri            = local.training_image
            ContainerEntrypoint = ["bash", "/opt/ml/processing/code/pipeline_eval.sh"]
          }
          ProcessingInputs = [
            {
              InputName = "code"
              S3Input   = { S3Uri = "${local.ml_s3}/code/pipeline/eval/", LocalPath = "/opt/ml/processing/code", S3DataType = "S3Prefix", S3InputMode = "File", S3DataDistributionType = "FullyReplicated" }
            },
            {
              InputName = "test"
              S3Input   = { S3Uri = "s3://${local.data_bucket}/processed/split=test/", LocalPath = "/opt/ml/processing/test", S3DataType = "S3Prefix", S3InputMode = "File", S3DataDistributionType = "FullyReplicated" }
            },
            {
              InputName = "model"
              S3Input   = { S3Uri = { Get = "Steps.Train.ModelArtifacts.S3ModelArtifacts" }, LocalPath = "/opt/ml/processing/model", S3DataType = "S3Prefix", S3InputMode = "File", S3DataDistributionType = "FullyReplicated" }
            },
            {
              InputName = "baseline"
              S3Input   = { S3Uri = { Get = "Parameters.BaselineMetricsUri" }, LocalPath = "/opt/ml/processing/baseline", S3DataType = "S3Prefix", S3InputMode = "File", S3DataDistributionType = "FullyReplicated" }
            },
          ]
          ProcessingOutputConfig = {
            Outputs = [{
              OutputName = "evaluation"
              S3Output = {
                S3Uri        = { "Std:Join" = { On = "/", Values = ["${local.ml_s3}/evaluation/pipeline", local.execution_id] } }
                LocalPath    = "/opt/ml/processing/evaluation"
                S3UploadMode = "EndOfJob"
              }
            }]
          }
          ProcessingResources = { ClusterConfig = { InstanceCount = 1, InstanceType = "ml.g4dn.xlarge", VolumeSizeInGB = 30 } }
          StoppingCondition   = { MaxRuntimeInSeconds = 2700 }
          Environment         = { HF_HOME = "/tmp/hf", PYTHONUNBUFFERED = "1" }
          Tags                = local.pipeline_tags
        }
        PropertyFiles = [{ PropertyFileName = "EvaluationReport", OutputName = "evaluation", FilePath = "evaluation.json" }]
      },
      {
        Name = "ChrfGate"
        Type = "Condition"
        Arguments = {
          Conditions = [{
            Type       = "GreaterThanOrEqualTo"
            LeftValue  = { "Std:JsonGet" = { PropertyFile = { Get = "Steps.Evaluate.PropertyFiles.EvaluationReport" }, Path = "gate.promote" } }
            RightValue = 1
          }]
          IfSteps = [{
            Name = "Register"
            Type = "RegisterModel"
            Arguments = {
              ModelPackageGroupName   = aws_sagemaker_model_package_group.translation.model_package_group_name
              ModelPackageDescription = "NLLB-200 600M fine-tuned on a subset of the English-Nuer-Dinka corpus by Tajnaam (CC BY-NC 4.0). Promoted by the chrF gate; metrics measure agreement with very probably machine-generated references."
              ModelApprovalStatus     = "Approved"
              ModelMetrics = {
                ModelQuality = {
                  Statistics = {
                    ContentType = "application/json"
                    S3Uri = { "Std:Join" = { On = "/", Values = [
                      { Get = "Steps.Evaluate.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri" },
                      "evaluation.json",
                    ] } }
                  }
                }
              }
              InferenceSpecification = {
                Containers = [{
                  Image        = local.inference_image
                  ModelDataUrl = { Get = "Steps.Train.ModelArtifacts.S3ModelArtifacts" }
                }]
                SupportedContentTypes                   = ["application/json"]
                SupportedResponseMIMETypes              = ["application/json"]
                SupportedRealtimeInferenceInstanceTypes = ["ml.g4dn.xlarge", "ml.m5.xlarge"]
              }
              CustomerMetadataProperties = {
                base_model          = "facebook/nllb-200-distilled-600M"
                base_model_revision = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
                dataset             = "English-Nuer-Dinka Parallel Corpus by Tajnaam - CC BY-NC 4.0 - revision 28db28e6"
                promotion_rule      = "fine-tuned chrF greater than baseline chrF in every direction"
              }
            }
          }]
          ElseSteps = [{
            Name      = "GateNotPassed"
            Type      = "Fail"
            Arguments = { ErrorMessage = "Fine-tuned chrF did not beat the baseline in every direction; the model was not registered." }
          }]
        }
      },
    ]
  }
}

# ---------------------------------------------------------------- Code used by the pipeline

resource "aws_s3_object" "train_sourcedir" {
  bucket = aws_s3_bucket.ml.id
  key    = "code/pipeline/train/sourcedir.tar.gz"
  source = local.sourcedir_path
  etag   = filemd5(local.sourcedir_path)
}

resource "aws_s3_object" "eval_code" {
  for_each = toset(local.eval_code_files)

  bucket = aws_s3_bucket.ml.id
  key    = "code/pipeline/eval/${each.value}"
  source = "${path.module}/../../evaluation/${each.value}"
  etag   = filemd5("${path.module}/../../evaluation/${each.value}")
}

# ---------------------------------------------------------------- Pipeline execution role

data "aws_iam_policy_document" "pipeline" {
  statement {
    sid     = "RunPipelineJobs"
    actions = ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob", "sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob", "sagemaker:AddTags"]
    resources = [
      "arn:${data.aws_partition.current.partition}:sagemaker:${local.region}:${local.account}:training-job/pipelines-*",
      "arn:${data.aws_partition.current.partition}:sagemaker:${local.region}:${local.account}:processing-job/pipelines-*",
    ]
  }

  statement {
    sid     = "RegisterModelVersion"
    actions = ["sagemaker:CreateModelPackage", "sagemaker:CreateModelPackageGroup", "sagemaker:DescribeModelPackage", "sagemaker:DescribeModelPackageGroup", "sagemaker:AddTags"]
    resources = [
      aws_sagemaker_model_package_group.translation.arn,
      "arn:${data.aws_partition.current.partition}:sagemaker:${local.region}:${local.account}:model-package/${aws_sagemaker_model_package_group.translation.model_package_group_name}/*",
    ]
  }

  statement {
    sid       = "PassStageRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.training.arn, aws_iam_role.evaluation.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  statement {
    sid       = "ReadReportsAndArtifacts"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.ml.arn}/evaluation/*", "${aws_s3_bucket.ml.arn}/training/*"]
  }
}

resource "aws_iam_role" "pipeline" {
  name               = "${local.name}-pipeline"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy" "pipeline" {
  name   = "pipeline"
  role   = aws_iam_role.pipeline.id
  policy = data.aws_iam_policy_document.pipeline.json
}

# ---------------------------------------------------------------- Pipeline

resource "aws_sagemaker_pipeline" "train_eval_register" {
  pipeline_name         = "${local.name}-train-eval-register"
  pipeline_display_name = "${local.name}-train-eval-register"
  pipeline_description  = "Fine-tune NLLB-200 600M (English to Dinka and Nuer), evaluate on the held-out test split, register only if chrF beats the baseline in every direction."
  role_arn              = aws_iam_role.pipeline.arn
  pipeline_definition   = jsonencode(local.pipeline_definition)

  depends_on = [aws_iam_role_policy.pipeline, aws_s3_object.train_sourcedir, aws_s3_object.eval_code]
}

output "pipeline_name" {
  value = aws_sagemaker_pipeline.train_eval_register.pipeline_name
}
