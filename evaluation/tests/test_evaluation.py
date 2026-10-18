import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "training" / "tests"))
import run_eval  # noqa: E402
import gate  # noqa: E402
from tiny import build_tiny_model, write_split  # noqa: E402


def report(chrf_dik, chrf_nus, **overrides):
    r = {
        "test_file_sha256": "abc",
        "test_rows": 100,
        "limited": False,
        "decoding": {"num_beams": 4, "max_new_tokens": 200},
        "directions": {
            "eng_Latn-dik_Latn": {"bleu": 5.0, "chrf": chrf_dik},
            "eng_Latn-nus_Latn": {"bleu": 4.0, "chrf": chrf_nus},
        },
    }
    r.update(overrides)
    return r


def test_gate_promotes_only_when_every_direction_improves():
    base = report(30.0, 25.0)
    assert gate.compare(base, report(31.0, 26.0))["gate"]["promote"] == 1
    assert gate.compare(base, report(31.0, 24.9))["gate"]["promote"] == 0
    assert gate.compare(base, report(30.0, 26.0))["gate"]["promote"] == 0  # equal is not better


def test_gate_refuses_different_settings_or_test_file():
    base = report(30.0, 25.0)
    for key, value in (("test_file_sha256", "other"), ("test_rows", 99), ("decoding", {"num_beams": 1, "max_new_tokens": 200}), ("limited", True)):
        with pytest.raises(ValueError):
            gate.compare(base, report(31.0, 26.0, **{key: value}))
    other = copy.deepcopy(base)
    del other["directions"]["eng_Latn-nus_Latn"]
    with pytest.raises(ValueError):
        gate.compare(base, other)


def test_evaluate_writes_metrics_with_signatures(tmp_path):
    model_dir = build_tiny_model(tmp_path / "tiny")
    test_dir = write_split(tmp_path / "test", 3)
    out = tmp_path / "eval"
    run_eval.main(["--model", str(model_dir), "--label", "base", "--test_dir", str(test_dir), "--out_dir", str(out),
                   "--num_beams", "2", "--max_new_tokens", "5", "--batch_size", "2"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_rows"] == 3 and metrics["label"] == "base"
    for d in ("eng_Latn-dik_Latn", "eng_Latn-nus_Latn"):
        m = metrics["directions"][d]
        assert m["sentences"] == 3
        assert "nrefs:1" in m["bleu_signature"] and "nrefs:1" in m["chrf_signature"]
    preds = (out / "predictions.jsonl").read_text().splitlines()
    assert len(preds) == 6
    # Gate end to end on two real reports from the same settings.
    result = gate.compare(metrics, metrics)
    assert result["gate"]["promote"] == 0
