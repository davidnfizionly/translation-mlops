"""Register the model of one pipeline execution whose Register step failed.

Takes the Register step arguments from the live pipeline definition, replaces
the step references with the real outputs of the execution's Train and
Evaluate jobs, adds the execution and job names as metadata, and calls
CreateModelPackage once. Train and Evaluate are not re-run. --dry-run prints
the request without creating anything.
"""

import argparse
import json
import re

import boto3

PIPELINE = "translation-mlops-train-eval-register"
# CreateModelPackage pattern for CustomerMetadataProperties values.
VALUE = re.compile(r"[\w .:/=+\-@]{1,256}")


def find_step(steps, name):
    for step in steps:
        if step["Name"] == name:
            return step
        if step["Type"] == "Condition":
            found = find_step(step["Arguments"]["IfSteps"] + step["Arguments"]["ElseSteps"], name)
            if found:
                return found
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--execution-id", required=True)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sm = boto3.Session(region_name=args.region).client("sagemaker")
    account = boto3.Session(region_name=args.region).client("sts").get_caller_identity()["Account"]
    execution_arn = f"arn:aws:sagemaker:{args.region}:{account}:pipeline/{PIPELINE}/execution/{args.execution_id}"
    steps = {s["StepName"]: s for s in sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)["PipelineExecutionSteps"]}
    for name in ("Train", "Evaluate", "ChrfGate"):
        assert steps[name]["StepStatus"] == "Succeeded", (name, steps[name]["StepStatus"])
    assert steps["ChrfGate"]["Metadata"]["Condition"]["Outcome"] == "True"
    train_job = steps["Train"]["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
    eval_job = steps["Evaluate"]["Metadata"]["ProcessingJob"]["Arn"].split("/")[-1]
    model_data = sm.describe_training_job(TrainingJobName=train_job)["ModelArtifacts"]["S3ModelArtifacts"]
    outputs = sm.describe_processing_job(ProcessingJobName=eval_job)["ProcessingOutputConfig"]["Outputs"]
    eval_uri = next(o["S3Output"]["S3Uri"] for o in outputs if o["OutputName"] == "evaluation").rstrip("/")

    definition = json.loads(sm.describe_pipeline(PipelineName=PIPELINE)["PipelineDefinition"])
    reg = find_step(definition["Steps"], "Register")["Arguments"]
    reg["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"] = f"{eval_uri}/evaluation.json"
    reg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = model_data
    reg["ModelPackageDescription"] += f" Registered from pipeline execution {args.execution_id} after its Register step failed."
    reg["CustomerMetadataProperties"].update({
        "pipeline_execution_id": args.execution_id,
        "training_job": train_job,
        "evaluation_job": eval_job,
        "registered_by": "scripts/register_model_version.py - Register step failed in the pipeline",
    })
    for key, value in reg["CustomerMetadataProperties"].items():
        assert VALUE.fullmatch(value), (key, value)
    assert "Get" not in json.dumps(reg), "unresolved step reference"
    # Versions take no tags (the API refuses them); the group carries Project=translation-mlops.
    request = reg

    if args.dry_run:
        print(re.sub(r"s3://[^/\"]+", "s3://<bucket>", json.dumps(request, indent=2)))
        return
    print(sm.create_model_package(**request)["ModelPackageArn"].split("/", 1)[1])


if __name__ == "__main__":
    main()
