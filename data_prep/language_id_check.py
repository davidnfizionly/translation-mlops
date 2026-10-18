"""Language identification of a random sample with GlotLID v3 (fastText).

GlotLID v3 separates Southwestern Dinka (dik_Latn, the variety NLLB-200
covers), Northeastern Dinka (dip_Latn) and Southeastern Dinka (dks_Latn).
This gives a measured hint of which Dinka variety the corpus is written in.
It is a classifier output, not a linguistic verification.

GlotLID: https://huggingface.co/cis-lmu/glotlid (Apache 2.0)
"""

import argparse
import json
import random
from collections import Counter

import fasttext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)
    random.seed(args.seed)
    sample = random.sample(rows, args.sample)
    model = fasttext.load_model(args.model)

    print(f"GlotLID v3, random sample of {args.sample} rows (seed {args.seed}) out of {len(rows)}")
    for field in ("din", "nuer", "english"):
        texts = [r[field].replace("\n", " ") for r in sample]
        labels, probs = model.predict(texts, k=1)
        top = Counter(l[0].replace("__label__", "") for l in labels)
        conf = [p[0] for p in probs]
        confident = sum(1 for p in conf if p >= 0.9)
        print(f"\nfield {field}: share with top-1 probability >= 0.9: {confident / len(conf):.1%}")
        for label, n in top.most_common(6):
            print(f"  {label:10} {n:5}  {n / len(texts):6.1%}")
        if field == "din":
            agg = Counter()
            labels5, probs5 = model.predict(texts, k=5)
            for ls, ps in zip(labels5, probs5):
                for l, p in zip(ls, ps):
                    l = l.replace("__label__", "")
                    if l in ("dik_Latn", "dip_Latn", "dks_Latn"):
                        agg[l] += p
            print("  mean probability mass per Dinka variety:")
            for l in ("dik_Latn", "dip_Latn", "dks_Latn"):
                print(f"    {l}: {agg[l] / len(texts):.3f}")


if __name__ == "__main__":
    main()
