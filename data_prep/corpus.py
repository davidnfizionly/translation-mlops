"""Cleaning, deduplicated split and leakage measures for the English-Nuer-Dinka corpus.

Standard library only, so the same module runs locally, in the tests and in
the Glue Python shell job. Rows are dicts with the keys english, nuer, din.

Cleaning rules, applied in this order; a removed row is counted under the
first rule that removes it:

  normalize          NFC, zero-width characters removed, whitespace collapsed (rows are modified, not removed)
  replacement_char   any field contains U+FFFD (broken encoding)
  empty_field        a field is empty after normalisation
  english_length     English has fewer than 3 or more than 100 words
  target_loop        Nuer or Dinka repeats a word trigram 3 or more times while English does not
  length_ratio       Nuer/English or Dinka/English character ratio outside 0.5 to 2.5
  duplicate_english  English already seen (normalised key)
  duplicate_target   Nuer or Dinka already seen (normalised key)

Split: rows are grouped by the English key with digits masked, so sentences
that differ only in numbers stay together. Groups are ordered by a SHA-256
hash of the key and the seed; the first groups fill the test split, the next
ones the validation split, the rest is training. Test and validation rows
whose word-trigram Jaccard similarity with any training row reaches the
threshold (any of the three fields) are then removed from evaluation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict

FIELDS = ("english", "nuer", "din")
TARGETS = ("nuer", "din")

RULES = (
    "replacement_char",
    "empty_field",
    "english_length",
    "target_loop",
    "length_ratio",
    "duplicate_english",
    "duplicate_target",
)

MIN_EN_WORDS = 3
MAX_EN_WORDS = 100
LOOP_REPEATS = 3
RATIO_MIN = 0.5
RATIO_MAX = 2.5
NEAR_DUP_JACCARD = 0.8

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_SPACES = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    return _SPACES.sub(" ", text).strip()


def norm_key(text: str) -> str:
    """Key for duplicate detection: casefolded, punctuation removed."""
    text = _PUNCT.sub("", unicodedata.normalize("NFC", text).casefold())
    return _SPACES.sub(" ", text).strip()


def digit_key(text: str) -> str:
    """norm_key with every number replaced by 0."""
    return _DIGITS.sub("0", norm_key(text))


def max_trigram_repeat(text: str) -> int:
    words = text.split()
    if len(words) < 3:
        return 0
    grams = Counter(zip(words, words[1:], words[2:]))
    return grams.most_common(1)[0][1]


def shingles(text: str) -> frozenset:
    """Word trigrams of the normalised key (unigrams for very short text)."""
    words = digit_key(text).split()
    if len(words) < 3:
        return frozenset((w,) for w in words)
    return frozenset(zip(words, words[1:], words[2:]))


def removal_reason(row: dict, seen_en: set, seen_tgt: dict) -> str | None:
    """Return the first rule that removes the row, or None. Updates the seen sets."""
    if any("�" in row[f] for f in FIELDS):
        return "replacement_char"
    if any(not row[f] for f in FIELDS):
        return "empty_field"
    n_words = len(row["english"].split())
    if n_words < MIN_EN_WORDS or n_words > MAX_EN_WORDS:
        return "english_length"
    if max_trigram_repeat(row["english"]) < LOOP_REPEATS and any(
        max_trigram_repeat(row[t]) >= LOOP_REPEATS for t in TARGETS
    ):
        return "target_loop"
    en_len = len(row["english"])
    if any(not RATIO_MIN <= len(row[t]) / en_len <= RATIO_MAX for t in TARGETS):
        return "length_ratio"
    key = norm_key(row["english"])
    if key in seen_en:
        return "duplicate_english"
    tgt_keys = {t: norm_key(row[t]) for t in TARGETS}
    if any(tgt_keys[t] in seen_tgt[t] for t in TARGETS):
        return "duplicate_target"
    seen_en.add(key)
    for t in TARGETS:
        seen_tgt[t].add(tgt_keys[t])
    return None


def clean(rows: list) -> tuple[list, dict]:
    """Apply the cleaning rules. Returns (kept rows with row_id, stats)."""
    stats = {"input_rows": len(rows), "normalized_rows_changed": 0, "removed": dict.fromkeys(RULES, 0)}
    seen_en, seen_tgt = set(), {t: set() for t in TARGETS}
    kept = []
    for i, raw in enumerate(rows):
        row = {f: normalize_text(raw.get(f) or "") for f in FIELDS}
        if any(row[f] != (raw.get(f) or "") for f in FIELDS):
            stats["normalized_rows_changed"] += 1
        reason = removal_reason(row, seen_en, seen_tgt)
        if reason:
            stats["removed"][reason] += 1
            continue
        kept.append({"row_id": i, **row})
    stats["kept_rows"] = len(kept)
    return kept, stats


def _hash_order(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def split(rows: list, test_size: int = 2000, valid_size: int = 2000, seed: int = 13) -> tuple[dict, dict]:
    """Group-aware deterministic split, then near-duplicate removal from evaluation splits."""
    groups = defaultdict(list)
    for row in rows:
        groups[digit_key(row["english"])].append(row)
    order = sorted(groups, key=lambda k: _hash_order(k, seed))

    parts = {"test": [], "valid": [], "train": []}
    for key in order:
        if len(parts["test"]) < test_size:
            target = "test"
        elif len(parts["valid"]) < valid_size:
            target = "valid"
        else:
            target = "train"
        parts[target].extend(groups[key])

    eval_rows = parts["test"] + parts["valid"]
    best = max_jaccard_to_reference(eval_rows, parts["train"])
    near = {r["row_id"] for r in eval_rows if best[r["row_id"]] >= NEAR_DUP_JACCARD}
    removed = {name: sum(1 for r in parts[name] if r["row_id"] in near) for name in ("test", "valid")}
    for name in ("test", "valid"):
        parts[name] = [r for r in parts[name] if r["row_id"] not in near]

    stats = {
        "seed": seed,
        "groups": len(groups),
        "near_duplicate_threshold": NEAR_DUP_JACCARD,
        "near_duplicates_removed": removed,
        "sizes": {name: len(rows_) for name, rows_ in parts.items()},
    }
    return parts, stats


def max_jaccard_to_reference(queries: list, reference: list) -> dict:
    """For each query row_id, the highest word-trigram Jaccard similarity with any
    reference row, over the three fields (same field compared with same field)."""
    best = {q["row_id"]: 0.0 for q in queries}
    for field in FIELDS:
        q_sh = {q["row_id"]: shingles(q[field]) for q in queries}
        index = defaultdict(list)
        for qid, sh in q_sh.items():
            for g in sh:
                index[g].append(qid)
        for ref in reference:
            r_sh = shingles(ref[field])
            shared = Counter()
            for g in r_sh:
                for qid in index.get(g, ()):
                    shared[qid] += 1
            for qid, c in shared.items():
                j = c / (len(q_sh[qid]) + len(r_sh) - c)
                if j > best[qid]:
                    best[qid] = j
    return best
