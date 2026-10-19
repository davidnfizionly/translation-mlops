"""Compare looping outputs of two evaluation runs and print fixed-rule samples.

A hypothesis loops when one word trigram appears 3 or more times while the
source does not loop (same rule as the baseline proof). Samples are not
hand-picked: rows are ranked by sentence chrF against the reference, and 5 rows
are drawn with a fixed seed from the top quarter ("good") and 5 from the
bottom quarter ("bad"), per direction.

Usage: python scripts/output_quality.py BASE_PREDICTIONS FINETUNED_PREDICTIONS [SAMPLES_JSON]
SAMPLES_JSON, when given, receives the printed samples for the local widget.
"""

import json
import random
import sys
from pathlib import Path

from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from corpus import max_trigram_repeat  # noqa: E402

SEED = 13
SAMPLES = 5


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def loops(row):
    return max_trigram_repeat(row["hypothesis"]) >= 3 and max_trigram_repeat(row["source"]) < 3


def main(base_path, ft_path, json_path=None):
    base, ft = load(base_path), load(ft_path)
    chrf = CHRF()
    for direction in sorted({r["direction"] for r in ft}):
        b = [r for r in base if r["direction"] == direction]
        f = [r for r in ft if r["direction"] == direction]
        nb, nf = sum(map(loops, b)), sum(map(loops, f))
        print(f"{direction}: looping outputs base {nb} of {len(b)} ({100 * nb / len(b):.1f}%), "
              f"fine-tuned {nf} of {len(f)} ({100 * nf / len(f):.1f}%); "
              f"empty base {sum(not r['hypothesis'].strip() for r in b)}, fine-tuned {sum(not r['hypothesis'].strip() for r in f)}")
    print()
    rng = random.Random(SEED)
    base_by_key = {(r["row_id"], r["direction"]): r for r in base}
    samples = []
    for direction in sorted({r["direction"] for r in ft}):
        rows = [r for r in ft if r["direction"] == direction]
        scored = sorted(((chrf.sentence_score(r["hypothesis"], [r["reference"]]).score, r) for r in rows), key=lambda x: (x[0], x[1]["row_id"]))
        quarter = len(scored) // 4
        for label, pool in (("good", scored[-quarter:]), ("bad", scored[:quarter])):
            for score, r in sorted(rng.sample(pool, SAMPLES), key=lambda x: x[1]["row_id"]):
                print(f"[{direction}] {label}, row {r['row_id']}, sentence chrF {score:.1f}, looping {loops(r)}")
                print(f"  source:     {r['source']}")
                print(f"  reference:  {r['reference']}")
                print(f"  fine-tuned: {r['hypothesis']}")
                print(f"  base:       {base_by_key[(r['row_id'], direction)]['hypothesis']}")
                print()
                samples.append({"row_id": r["row_id"], "direction": direction, "label": label, "sentence_chrf": round(score, 1),
                                "looping": loops(r), "source": r["source"], "reference": r["reference"],
                                "finetuned": r["hypothesis"], "base": base_by_key[(r["row_id"], direction)]["hypothesis"]})
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
