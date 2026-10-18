"""Check that no test or validation sentence leaks into training.

Checks, for each pair of splits (test-train, valid-train, test-valid):
  1. exact match of the normalised key, per field (english, nuer, din)
  2. exact match of the English key with digits masked
  3. near duplicates: word-trigram Jaccard >= threshold, per field

Exits with status 1 when any count is not zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402


def load(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check(queries: list, reference: list, threshold: float) -> dict:
    result = {}
    for field in corpus.FIELDS:
        ref_keys = {corpus.norm_key(r[field]) for r in reference}
        result[f"exact_{field}"] = sum(1 for q in queries if corpus.norm_key(q[field]) in ref_keys)
    ref_digit = {corpus.digit_key(r["english"]) for r in reference}
    result["digit_masked_english"] = sum(1 for q in queries if corpus.digit_key(q["english"]) in ref_digit)
    best = corpus.max_jaccard_to_reference(queries, reference)
    result[f"near_duplicate_jaccard_ge_{threshold}"] = sum(1 for v in best.values() if v >= threshold)
    result["max_jaccard"] = round(max(best.values(), default=0.0), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="data/processed")
    parser.add_argument("--threshold", type=float, default=corpus.NEAR_DUP_JACCARD)
    args = parser.parse_args()

    d = Path(args.dir)
    splits = {name: load(d / f"{name}.jsonl") for name in ("train", "valid", "test")}
    print("Split sizes: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    ids = {name: {r["row_id"] for r in rows} for name, rows in splits.items()}
    shared_ids = len(ids["train"] & ids["test"]) + len(ids["train"] & ids["valid"]) + len(ids["test"] & ids["valid"])
    print(f"Row ids shared between splits: {shared_ids}")

    failures = shared_ids
    for q, r in (("test", "train"), ("valid", "train"), ("test", "valid")):
        result = check(splits[q], splits[r], args.threshold)
        print(f"{q} vs {r}:")
        for k, v in result.items():
            print(f"  {k:32} {v}")
            if k != "max_jaccard":
                failures += v
    print("RESULT: PASS (0 leaks)" if failures == 0 else f"RESULT: FAIL ({failures} leaks)")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
