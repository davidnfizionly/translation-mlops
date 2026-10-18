"""Package training/ and start one SageMaker training job with boto3.

The Hugging Face training image runs train.py through the SageMaker training
toolkit (sagemaker_program / sagemaker_submit_directory hyperparameters).
Prints the job name; --dry-run prints the request without creating anything.

Bucket and role names are read from `terraform output` of infra/ml, so no
account-specific value is stored in the repository.
"""

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[1]
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"

PROFILES = {
    # Validates the script and the container: few rows, few steps.
    "smoke": {"train_rows": 2000, "valid_rows": 100, "max_steps": 30, "eval_steps": 15, "save_steps": 15, "warmup_steps": 5},
}

METRICS = [
    {"Name": "train:loss", "Regex": r"'loss': ([0-9.]+)"},
    {"Name": "validation:loss", "Regex": r"'eval_loss': ([0-9.]+)"},
    {"Name": "train:samples_per_second", "Regex": r"'train_samples_per_second': ([0-9.]+)"},
]


def tf_output(stack):
    out = subprocess.run(["terraform", f"-chdir={REPO / 'infra' / stack}", "output", "-json"], capture_output=True, text=True, check=True)
    return {k: v["value"] for k, v in json.loads(out.stdout).items()}


def source_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in ["train.py", "serving/inference.py"]:
            tar.add(REPO / "training" / path, arcname=path)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--max-runtime-seconds", type=int, default=1800)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ml = tf_output("ml")
    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/{ml['training_role_name']}"
    job = f"tm-{args.profile}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    code_key = f"code/{job}/sourcedir.tar.gz"

    hyper = {k: str(v) for k, v in PROFILES[args.profile].items()}
    hyper.update({
        "sagemaker_program": json.dumps("train.py"),
        "sagemaker_submit_directory": json.dumps(f"s3://{ml['ml_bucket']}/{code_key}"),
        "sagemaker_region": json.dumps(args.region),
    })
    request = {
        "TrainingJobName": job,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE.format(region=args.region),
            "TrainingInputMode": "File",
            "MetricDefinitions": METRICS,
        },
        "HyperParameters": hyper,
        "InputDataConfig": [
            {
                "ChannelName": name,
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{ml['data_bucket']}/processed/split={name}/", "S3DataDistributionType": "FullyReplicated"}},
                "ContentType": "application/x-ndjson",
            }
            for name in ("train", "valid")
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{ml['ml_bucket']}/training/"},
        "CheckpointConfig": {"S3Uri": f"s3://{ml['ml_bucket']}/checkpoints/{job}/"},
        "ResourceConfig": {"InstanceType": args.instance_type, "InstanceCount": 1, "VolumeSizeInGB": 50},
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "Tags": [{"Key": "Project", "Value": "translation-mlops"}, {"Key": "Run", "Value": args.profile}],
    }
    if args.dry_run:
        shown = json.loads(json.dumps(request))
        shown["RoleArn"] = "arn:aws:iam::<account>:role/" + ml["training_role_name"]
        print(json.dumps(shown, indent=2))
        return

    s3 = session.client("s3")
    s3.put_object(Bucket=ml["ml_bucket"], Key=code_key, Body=source_tarball())
    session.client("sagemaker").create_training_job(**request)
    print(job)


if __name__ == "__main__":
    sys.exit(main())
