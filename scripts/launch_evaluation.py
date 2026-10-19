"""Start one SageMaker processing job that runs evaluation/run_eval.py on the test split.

--model is a hub id (baseline) or --model-s3 is a training job's model.tar.gz
prefix (fine-tuned). Decoding settings are the defaults of run_eval.py, the
same for every model. Prints the job name; --dry-run prints the request only.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[1]
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
BASE_MODEL = "facebook/nllb-200-distilled-600M"
BASE_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def tf_output(stack):
    out = subprocess.run(["terraform", f"-chdir={REPO / 'infra' / stack}", "output", "-json"], capture_output=True, text=True, check=True)
    return {k: v["value"] for k, v in json.loads(out.stdout).items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True, help="base or finetuned")
    p.add_argument("--model-s3", help="S3 prefix holding model.tar.gz (fine-tuned); default: the base model from the hub")
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--max-runtime-seconds", type=int, default=2700)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ml = tf_output("ml")
    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    job = f"tm-eval-{args.label}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    code_prefix = f"code/{job}/"

    if args.model_s3:
        arguments = ["--model", "/opt/ml/processing/model"]
    else:
        arguments = ["--model", BASE_MODEL, "--model_revision", BASE_REVISION]
    arguments += ["--label", args.label]
    inputs = [
        {"InputName": "code", "S3Input": {"S3Uri": f"s3://{ml['ml_bucket']}/{code_prefix}", "LocalPath": "/opt/ml/processing/code", "S3DataType": "S3Prefix", "S3InputMode": "File"}},
        {"InputName": "test", "S3Input": {"S3Uri": f"s3://{ml['data_bucket']}/processed/split=test/", "LocalPath": "/opt/ml/processing/test", "S3DataType": "S3Prefix", "S3InputMode": "File"}},
    ]
    if args.model_s3:
        inputs.append({"InputName": "model", "S3Input": {"S3Uri": args.model_s3, "LocalPath": "/opt/ml/processing/model", "S3DataType": "S3Prefix", "S3InputMode": "File"}})

    request = {
        "ProcessingJobName": job,
        "RoleArn": f"arn:aws:iam::{account}:role/{ml['evaluation_role_name']}",
        "AppSpecification": {
            "ImageUri": IMAGE.format(region=args.region),
            "ContainerEntrypoint": ["bash", "/opt/ml/processing/code/run_eval.sh"],
            "ContainerArguments": arguments,
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {
            "Outputs": [
                {"OutputName": "evaluation", "S3Output": {"S3Uri": f"s3://{ml['ml_bucket']}/evaluation/{job}/", "LocalPath": "/opt/ml/processing/evaluation", "S3UploadMode": "EndOfJob"}}
            ]
        },
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": args.instance_type, "VolumeSizeInGB": 30}},
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "Environment": {"HF_HOME": "/tmp/hf", "PYTHONUNBUFFERED": "1"},
        "Tags": [{"Key": "Project", "Value": "translation-mlops"}, {"Key": "Run", "Value": f"eval-{args.label}"}],
    }
    if args.dry_run:
        shown = json.loads(json.dumps(request))
        shown["RoleArn"] = "arn:aws:iam::<account>:role/" + ml["evaluation_role_name"]
        print(json.dumps(shown, indent=2))
        return

    s3 = session.client("s3")
    for name in ("run_eval.py", "run_eval.sh", "requirements.txt"):
        s3.upload_file(str(REPO / "evaluation" / name), ml["ml_bucket"], code_prefix + name)
    session.client("sagemaker").create_processing_job(**request)
    print(job)


if __name__ == "__main__":
    main()
