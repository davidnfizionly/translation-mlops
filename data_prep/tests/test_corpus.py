import json
import subprocess
import sys
from pathlib import Path

import pytest

DATA_PREP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_PREP))
import corpus  # noqa: E402


def row(en, nuer=None, din=None):
    return {"english": en, "nuer": nuer if nuer is not None else en + " nu", "din": din if din is not None else en + " di"}


def test_normalize_removes_zero_width_and_collapses_spaces():
    assert corpus.normalize_text("  a​ b \t c ") == "a b c"


def test_normalize_composes_nfc():
    assert corpus.normalize_text("ë") == "ë"


def test_norm_key_ignores_case_and_punctuation():
    assert corpus.norm_key("Hello, World!") == corpus.norm_key("hello world")


def test_digit_key_masks_numbers():
    assert corpus.digit_key("Won 106 medals in 2015") == corpus.digit_key("won 12 medals in 1999")


@pytest.mark.parametrize(
    "r, rule",
    [
        (row("one two three", din="bad � text here"), "replacement_char"),
        (row("one two three", nuer=""), "empty_field"),
        (row("too short"), "english_length"),
        (row(" ".join(["w"] * 101)), "english_length"),
        (row("one two three four", din="ke tou ke tou ke tou ke tou ke"), "target_loop"),
        (row("one two three four five six", din="x"), "length_ratio"),
    ],
)
def test_each_rule_removes_its_case(r, rule):
    kept, stats = corpus.clean([r])
    assert kept == []
    assert stats["removed"][rule] == 1


def test_loop_kept_when_english_repeats_too():
    en = "go go go go go now"
    kept, _ = corpus.clean([row(en, nuer=en, din=en)])
    assert len(kept) == 1


def test_duplicates_keep_first_occurrence():
    rows = [
        row("The cat sat down.", "aaaa bbbb cccc dd", "eeee ffff gggg hh"),
        row("the cat sat down", "iiii jjjj kkkk ll", "mmmm nnnn oooo pp"),
        row("Another sentence here.", "aaaa bbbb cccc dd", "qqqq rrrr ssss tttt uu"),
    ]
    kept, stats = corpus.clean(rows)
    assert [r["row_id"] for r in kept] == [0]
    assert stats["removed"]["duplicate_english"] == 1
    assert stats["removed"]["duplicate_target"] == 1


def synthetic(n):
    return [
        {"row_id": i, "english": f"sentence {i} about topic {i * 7} here", "nuer": f"nuer {i} word", "din": f"din {i} word"}
        for i in range(n)
    ]


def test_split_is_deterministic_and_disjoint():
    rows = [
        {"row_id": i, "english": f"alpha{i} beta{i} gamma{i} delta", "nuer": f"n{i} m{i} o{i}", "din": f"d{i} e{i} f{i}"}
        for i in range(300)
    ]
    a, _ = corpus.split(rows, 20, 20, seed=1)
    b, _ = corpus.split(rows, 20, 20, seed=1)
    assert a == b
    ids = [{r["row_id"] for r in a[k]} for k in ("train", "valid", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(s) for s in ids) == 300


def test_number_variants_stay_in_one_split():
    rows = synthetic(200)  # English differs only in numbers: one group
    parts, stats = corpus.split(rows, 5, 5, seed=1)
    assert stats["groups"] == 1
    assert sum(1 for k in parts if parts[k]) == 1


def test_near_duplicate_removed_from_test():
    base = [
        {"row_id": i, "english": f"w{i}a w{i}b w{i}c w{i}d w{i}e", "nuer": f"n{i}a n{i}b n{i}c", "din": f"d{i}a d{i}b d{i}c"}
        for i in range(100)
    ]
    parts, _ = corpus.split(base, 10, 0, seed=3)
    victim = parts["test"][0]
    # A training row whose Dinka equals the test Dinka (other fields unrelated).
    leak = {"row_id": 999, "english": "unrelated words entirely different here", "nuer": "zz yy xx", "din": victim["din"] + " "}
    best = corpus.max_jaccard_to_reference([victim], [leak])
    assert best[victim["row_id"]] == 1.0


def run(cmd, cwd):
    return subprocess.run([sys.executable, "-I", *cmd], cwd=cwd, capture_output=True, text=True)


def test_leakage_check_passes_and_fails(tmp_path):
    rows = [
        row(f"alpha{i} beta{i} gamma{i} delta{i}", f"n{i} m{i} o{i} p{i}", f"d{i} e{i} f{i} g{i}") for i in range(120)
    ]
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(rows))
    out = tmp_path / "processed"
    r = run([str(DATA_PREP / "prepare.py"), "--input", str(raw), "--out", str(out), "--test-size", "10", "--valid-size", "10"], tmp_path)
    assert r.returncode == 0, r.stderr
    ok = run([str(DATA_PREP / "leakage_check.py"), "--dir", str(out)], tmp_path)
    assert ok.returncode == 0 and "PASS" in ok.stdout, ok.stdout + ok.stderr

    # Plant a leak: copy one test row into train.
    test_line = (out / "test.jsonl").read_text().splitlines()[0]
    planted = json.loads(test_line)
    planted["row_id"] = 10_000
    with (out / "train.jsonl").open("a") as f:
        f.write(json.dumps(planted) + "\n")
    bad = run([str(DATA_PREP / "leakage_check.py"), "--dir", str(out)], tmp_path)
    assert bad.returncode == 1 and "FAIL" in bad.stdout
