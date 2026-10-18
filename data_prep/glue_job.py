"""AWS Glue Python shell job (Python 3.9): clean, split and leakage-check the corpus.

Reads the raw JSON from S3, applies the same corpus module as the local run
(shipped as a wheel through --extra-py-files), writes JSON Lines per split
under processed/split=<name>/, the statistics and the leakage report under
reports/, and fails the job if the leakage check finds any leak.

Job arguments: --bucket, --raw_key, --seed, --test_size, --valid_size
"""

import json
import sys
import tempfile

import boto3
from awsglue.utils import getResolvedOptions

import corpus
import leakage_check

args = getResolvedOptions(sys.argv, ["bucket", "raw_key", "seed", "test_size", "valid_size"])
bucket = args["bucket"]
s3 = boto3.client("s3")


def put_text(key, text, content_type="application/json"):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


with tempfile.NamedTemporaryFile(suffix=".json") as raw:
    s3.download_fileobj(bucket, args["raw_key"], raw)
    raw.seek(0)
    rows = json.load(raw)

kept, clean_stats = corpus.clean(rows)
del rows
parts, split_stats = corpus.split(kept, int(args["test_size"]), int(args["valid_size"]), int(args["seed"]))
del kept

for name, split_rows in parts.items():
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows)
    put_text(f"processed/split={name}/{name}.jsonl", body, "application/x-ndjson")

leaks = {}
failures = 0
for q, ref in (("test", "train"), ("valid", "train"), ("test", "valid")):
    result = leakage_check.check(parts[q], parts[ref], corpus.NEAR_DUP_JACCARD)
    leaks[f"{q}_vs_{ref}"] = result
    failures += sum(v for k, v in result.items() if k != "max_jaccard")

stats = {"cleaning": clean_stats, "split": split_stats, "leakage": leaks, "leaks_total": failures}
put_text("reports/stats.json", json.dumps(stats, indent=2) + "\n")
print(json.dumps(stats, indent=2))

if failures:
    raise SystemExit(f"leakage check failed: {failures} leaks")
