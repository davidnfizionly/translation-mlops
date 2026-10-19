#!/bin/bash
# Runs the pipeline Evaluate step (evaluation/pipeline_eval.sh) locally in the
# training image, with the SageMaker processing paths, a tiny NLLB-architecture
# model packed as model.tar.gz, and a baseline report from the same tiny model.
# Expected: evaluation.json is written and gate.promote is 0 (no improvement).
# Usage: pipelines/tests/eval_step_in_container.sh <work dir outside the repository>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$1"
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04
mkdir -p "$WORK"
docker run --rm --entrypoint bash \
  -v "$REPO":/repo:ro -v "$REPO/evaluation":/opt/ml/processing/code:ro -v "$WORK":/work \
  -e HF_HOME=/work/hf "$IMAGE" -c '
set -euo pipefail
pip install -q -r /opt/ml/processing/code/requirements.txt
cd /work && rm -rf tiny proc && mkdir -p proc
python - <<PY
import sys, tarfile
sys.path.insert(0, "/repo/training/tests")
from tiny import build_tiny_model, write_split
build_tiny_model("/work/tiny")
write_split("/opt/ml/processing/test", 3)
import os; os.makedirs("/opt/ml/processing/model", exist_ok=True)
with tarfile.open("/opt/ml/processing/model/model.tar.gz", "w:gz") as tar:
    tar.add("/work/tiny", arcname=".")
PY
python /opt/ml/processing/code/run_eval.py --model /work/tiny --label base --test_dir /opt/ml/processing/test --out_dir /opt/ml/processing/baseline --num_beams 4 --max_new_tokens 200 > /dev/null
bash /opt/ml/processing/code/pipeline_eval.sh > /work/step.log
python - <<PY
import json
r = json.load(open("/opt/ml/processing/evaluation/evaluation.json"))
assert r["gate"]["promote"] == 0, r
assert set(r["directions"]) == {"eng_Latn-dik_Latn", "eng_Latn-nus_Latn"}
print("EVAL STEP OK: evaluation.json written, gate.promote =", r["gate"]["promote"], "| files:", sorted(__import__("os").listdir("/opt/ml/processing/evaluation")))
PY
'
