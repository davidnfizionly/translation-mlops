"""Inspect the raw corpus: size, fields, empty values, duplicates, lengths,
copy-through, degenerate repetitions and personal data patterns.

Prints a plain-text report (meant for proof/). Reads only; writes nothing
except the report when --report is given.
"""

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

LANGS = ("english", "nuer", "din")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL = re.compile(r"(https?://|www\.)\S+", re.I)
# Phone-like: optional +, 9 to 15 digits with separators, not a year range or a plain number.
PHONE = re.compile(r"(?<![\w.])(\+\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]){2,4}\d{2,4}(?![\w.])")
HANDLE = re.compile(r"(?<![\w.])@[A-Za-z0-9_]{3,}")
HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def norm(text: str) -> str:
    """Normalisation used for duplicate detection."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def degenerate(text: str) -> bool:
    """True when a word trigram repeats 3 times or more in one sentence."""
    words = text.split()
    grams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    return bool(grams) and grams.most_common(1)[0][1] >= 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/nuer_dinka_translations_paired.json")
    parser.add_argument("--report", help="write the report to this path")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = []
    say = out.append

    say(f"Input: {args.input}")
    say(f"Top-level type: {type(rows).__name__}, rows: {len(rows)}")

    key_sets = Counter(tuple(sorted(r.keys())) for r in rows)
    say("Key sets: " + "; ".join(f"{k} x{v}" for k, v in key_sets.most_common()))
    types = Counter((k, type(v).__name__) for r in rows for k, v in r.items())
    say("Value types: " + "; ".join(f"{k}:{t} x{n}" for (k, t), n in sorted(types.items())))

    say("")
    say("Missing or empty values (after strip):")
    for lang in LANGS:
        missing = sum(1 for r in rows if lang not in r or r[lang] is None)
        empty = sum(1 for r in rows if isinstance(r.get(lang), str) and not r[lang].strip())
        say(f"  {lang}: missing {missing}, empty {empty}")
    complete = [r for r in rows if all(isinstance(r.get(l), str) and r[l].strip() for l in LANGS)]
    say(f"Rows with all three fields non-empty: {len(complete)}")

    say("")
    say("Duplicates (normalised: NFC, casefold, punctuation removed, spaces collapsed):")
    triples = Counter(tuple(norm(r[l]) for l in LANGS) for r in complete)
    say(f"  distinct triples: {len(triples)}, duplicate triple rows: {len(complete) - len(triples)}")
    for lang in LANGS:
        c = Counter(norm(r[lang]) for r in complete)
        say(f"  {lang}: distinct {len(c)}, rows sharing a value with another row: {sum(v for v in c.values() if v > 1)}")
    for src, tgt in (("english", "nuer"), ("english", "din")):
        pairs = Counter((norm(r[src]), norm(r[tgt])) for r in complete)
        say(f"  pair {src}-{tgt}: distinct {len(pairs)}, duplicate pair rows {len(complete) - len(pairs)}")
    eng_targets = {}
    for r in complete:
        eng_targets.setdefault(norm(r["english"]), set()).add(norm(r["din"]))
    conflicting = sum(1 for v in eng_targets.values() if len(v) > 1)
    say(f"  English sentences with more than one distinct Dinka translation: {conflicting}")

    say("")
    say("Lengths in words (p1 / p50 / p99 / max):")
    for lang in LANGS:
        w = [len(r[lang].split()) for r in complete]
        say(f"  {lang}: {pct(w, .01)} / {pct(w, .5)} / {pct(w, .99)} / {max(w)}  mean {statistics.mean(w):.1f}")
    for lang in ("nuer", "din"):
        ratios = [len(r[lang]) / max(1, len(r["english"])) for r in complete]
        say(
            f"  char length ratio {lang}/english: p1 {pct(ratios, .01):.2f}, p50 {pct(ratios, .5):.2f}, "
            f"p99 {pct(ratios, .99):.2f}, outside 0.5-2.5: {sum(1 for x in ratios if x < .5 or x > 2.5)}"
        )

    say("")
    say("Translation quality signals:")
    for lang in ("nuer", "din"):
        same = sum(1 for r in complete if norm(r[lang]) == norm(r["english"]))
        deg = sum(1 for r in complete if degenerate(r[lang]))
        say(f"  {lang}: identical to English {same}, degenerate repetition (trigram x3+) {deg}")
    deg_en = sum(1 for r in complete if degenerate(r["english"]))
    say(f"  english: degenerate repetition {deg_en} (baseline for the source side)")
    nn = Counter()
    for r in complete:
        en_nums = Counter(NUMBER.findall(r["english"]))
        if not en_nums:
            continue
        nn["with_numbers"] += 1
        for lang in ("nuer", "din"):
            if Counter(NUMBER.findall(r[lang])) == en_nums:
                nn[lang] += 1
    if nn["with_numbers"]:
        say(
            f"  rows with numbers in English: {nn['with_numbers']}; numbers copied exactly: "
            f"nuer {nn['nuer'] / nn['with_numbers']:.1%}, din {nn['din'] / nn['with_numbers']:.1%}"
        )
    nuer_eq_din = sum(1 for r in complete if norm(r["nuer"]) == norm(r["din"]))
    say(f"  Nuer identical to Dinka: {nuer_eq_din}")

    say("")
    say("Characters outside Latin and common punctuation (top 15 per language):")
    for lang in ("nuer", "din"):
        chars = Counter()
        for r in complete:
            for ch in unicodedata.normalize("NFC", r[lang]):
                if ord(ch) > 0x24F and not unicodedata.category(ch).startswith(("P", "Z", "M")):
                    chars[ch] += 1
        say(f"  {lang}: " + " ".join(f"{c}(U+{ord(c):04X}) x{n}" for c, n in chars.most_common(15)))

    say("")
    say("Personal data patterns (counts of rows with a match, any field):")
    for name, pat in (("e-mail", EMAIL), ("URL", URL), ("phone-like", PHONE), ("@handle", HANDLE), ("honorific + name", HONORIFIC)):
        hits = [r for r in complete if any(pat.search(r[l]) for l in LANGS)]
        say(f"  {name}: {len(hits)}")
    caps = sum(1 for r in complete if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", r["english"][1:]))
    say(f"  English rows with a capitalised two-word sequence after the first word (names, places, titles): {caps}")

    random.seed(args.seed)
    say("")
    say(f"Random sample (seed {args.seed}), 5 rows:")
    for r in random.sample(complete, 5):
        for lang in LANGS:
            say(f"  {lang:7} {r[lang]}")
        say("")

    text = "\n".join(out) + "\n"
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
