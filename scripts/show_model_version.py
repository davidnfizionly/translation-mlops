"""Print one Model Registry version in a readable form, with identifiers masked.

Account IDs, ARNs and bucket names are replaced by XXXX so the output can be
captured. Shows status, approval, metrics (with the scores read from the
referenced evaluation.json), metadata and the source pipeline execution.
"""

import argparse
import json
import re
from datetime import timezone

import boto3

GROUP = "translation-mlops-nllb-en-din-nus"


def mask(text):
    text = re.sub(r"arn:aws[\w-]*:[^\s\"']+", "arn:XXXX", text)
    text = re.sub(r"s3://[^/\s\"']+", "s3://XXXX", text)
    return re.sub(r"\b\d{12}\b", "XXXX", text)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    session = boto3.Session(region_name=args.region)
    sm, s3 = session.client("sagemaker"), session.client("s3")
    summaries = sm.list_model_packages(ModelPackageGroupName=GROUP)["ModelPackageSummaryList"]
    arn = next(s["ModelPackageArn"] for s in summaries if s["ModelPackageVersion"] == args.version)
    mp = sm.describe_model_package(ModelPackageName=arn)
    container = mp["InferenceSpecification"]["Containers"][0]
    stats_uri = mp["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    bucket, key = stats_uri[5:].split("/", 1)
    report = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    meta = mp["CustomerMetadataProperties"]

    lines = [
        f"Model package group : {mp['ModelPackageGroupName']}",
        f"Version             : {mp['ModelPackageVersion']}",
        f"ARN                 : {arn}",
        f"Approval status     : {mp['ModelApprovalStatus']}",
        f"Package status      : {mp['ModelPackageStatus']}",
        f"Created (UTC)       : {mp['CreationTime'].astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Model",
        f"  artifact          : {container['ModelDataUrl']}",
        f"  inference image   : {container['Image'].split('/', 1)[1]}",
        f"  instance types    : {', '.join(mp['InferenceSpecification']['SupportedRealtimeInferenceInstanceTypes'])}",
        "",
        "Model quality metrics",
        f"  statistics file   : {stats_uri}",
        f"  test rows         : {report['test_rows']} per direction",
        f"  decoding          : beam {report['decoding']['num_beams']}, max_new_tokens {report['decoding']['max_new_tokens']}, {report['decoding']['dtype']}",
        "  direction            base BLEU  tuned BLEU  base chrF  tuned chrF",
    ]
    for d, m in report["directions"].items():
        lines.append(f"  {d:<20} {m['base_bleu']:>9.2f}  {m['finetuned_bleu']:>10.2f}  {m['base_chrf']:>9.2f}  {m['finetuned_chrf']:>10.2f}")
    lines += [
        f"  gate              : {report['rule']} - promote {report['gate']['promote']}",
        "",
        "Source pipeline execution",
        f"  pipeline          : translation-mlops-train-eval-register",
        f"  execution id      : {meta['pipeline_execution_id']}",
        f"  training job      : {meta['training_job']}",
        f"  evaluation job    : {meta['evaluation_job']}",
        f"  registered by     : {meta['registered_by']}",
        "",
        "Metadata",
        f"  base model        : {meta['base_model']} @ {meta['base_model_revision'][:12]}",
        f"  dataset           : {meta['dataset']}",
        f"  promotion rule    : {meta['promotion_rule']}",
    ]
    print(mask("\n".join(lines)))


if __name__ == "__main__":
    main()
