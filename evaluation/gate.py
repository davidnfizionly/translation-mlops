"""Promotion gate: compare the fine-tuned model with the baseline on the test split.

The baseline metrics come from the same run_eval.py run on the base model.
The comparison is refused if the two reports used a different test file, a
different number of rows, or different decoding settings.

Rule (fixed before any fine-tuned result was seen): promote only if the
fine-tuned chrF is higher than the baseline chrF in every direction.

Writes evaluation.json, read by the pipeline condition step (gate.promote is 1 or 0).
"""

import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare(base, candidate):
    for key in ("test_file_sha256", "test_rows", "decoding", "limited"):
        if base[key] != candidate[key]:
            raise ValueError(f"reports are not comparable: {key} differs ({base[key]} vs {candidate[key]})")
    if set(base["directions"]) != set(candidate["directions"]):
        raise ValueError("reports cover different directions")

    directions = {}
    for d in sorted(base["directions"]):
        b, c = base["directions"][d], candidate["directions"][d]
        directions[d] = {
            "base_bleu": b["bleu"],
            "finetuned_bleu": c["bleu"],
            "delta_bleu": round(c["bleu"] - b["bleu"], 2),
            "base_chrf": b["chrf"],
            "finetuned_chrf": c["chrf"],
            "delta_chrf": round(c["chrf"] - b["chrf"], 2),
            "chrf_improved": c["chrf"] > b["chrf"],
        }
    promote = all(v["chrf_improved"] for v in directions.values())
    return {
        "rule": "promote if fine-tuned chrF > baseline chrF in every direction",
        "test_rows": base["test_rows"],
        "test_file_sha256": base["test_file_sha256"],
        "decoding": base["decoding"],
        "directions": directions,
        "gate": {"promote": 1 if promote else 0},
        "note": "References are very probably machine generated (not confirmed by the dataset author); "
        "scores measure agreement with them, not the real quality for a speaker.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", default="/opt/ml/processing/baseline")
    p.add_argument("--candidate", default="/opt/ml/processing/evaluation")
    p.add_argument("--out", default="/opt/ml/processing/evaluation/evaluation.json")
    args = p.parse_args(argv)
    result = compare(load(args.baseline), load(args.candidate))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
