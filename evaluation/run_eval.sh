#!/bin/bash
# Entry point of the evaluation processing job: install the scorer, then run
# run_eval.py on the mounted test split with the arguments given to the job.
set -euo pipefail
pip install -q -r /opt/ml/processing/code/requirements.txt
exec python /opt/ml/processing/code/run_eval.py \
  --test_dir /opt/ml/processing/test --out_dir /opt/ml/processing/evaluation "$@"
