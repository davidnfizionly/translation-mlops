"""Clean the raw corpus and write the deduplicated train, validation and test splits.

Input:  data/raw/nuer_dinka_translations_paired.json (from download.py)
Output: <out>/{train,valid,test}.jsonl and <out>/stats.json, plus a text report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402


def write_jsonl(path: Path, rows: list) -> str:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(clean_stats: dict, split_stats: dict, hashes: dict) -> str:
    lines = [f"Input rows: {clean_stats['input_rows']}"]
    lines.append(f"Rows modified by normalisation (NFC, zero-width removed, spaces): {clean_stats['normalized_rows_changed']}")
    lines.append("Rows removed per rule (first matching rule, in order):")
    for rule in corpus.RULES:
        lines.append(f"  {rule:18} {clean_stats['removed'][rule]:7}")
    lines.append(f"  {'total':18} {sum(clean_stats['removed'].values()):7}")
    lines.append(f"Rows kept: {clean_stats['kept_rows']}")
    lines.append("")
    lines.append(f"Split seed {split_stats['seed']}, {split_stats['groups']} groups (English key with digits masked)")
    nd = split_stats["near_duplicates_removed"]
    lines.append(
        f"Near duplicates of training rows removed (trigram Jaccard >= {split_stats['near_duplicate_threshold']}): "
        f"test {nd['test']}, valid {nd['valid']}"
    )
    for name in ("train", "valid", "test"):
        lines.append(f"  {name:5} {split_stats['sizes'][name]:7} rows  sha256 {hashes[name]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/raw/nuer_dinka_translations_paired.json")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    kept, clean_stats = corpus.clean(rows)
    parts, split_stats = corpus.split(kept, args.test_size, args.valid_size, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: write_jsonl(out / f"{name}.jsonl", parts[name]) for name in ("train", "valid", "test")}
    stats = {"cleaning": clean_stats, "split": split_stats, "sha256": hashes}
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(report(clean_stats, split_stats, hashes), end="")


if __name__ == "__main__":
    main()
