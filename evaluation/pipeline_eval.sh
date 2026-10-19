#!/bin/bash
# Entry point of the pipeline Evaluate step: score the fine-tuned model on the
# test split with the fixed decoding settings, then apply the promotion gate
# against the stored baseline report. Writes evaluation.json for the condition step.
set -euo pipefail
pip install -q -r /opt/ml/processing/code/requirements.txt
python /opt/ml/processing/code/run_eval.py \
  --model /opt/ml/processing/model --label finetuned \
  --test_dir /opt/ml/processing/test --out_dir /opt/ml/processing/evaluation
python /opt/ml/processing/code/gate.py \
  --baseline /opt/ml/processing/baseline \
  --candidate /opt/ml/processing/evaluation \
  --out /opt/ml/processing/evaluation/evaluation.json
