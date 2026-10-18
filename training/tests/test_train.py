import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "serving"))
import inference  # noqa: E402
import train  # noqa: E402
from tiny import build_tiny_model, write_split  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny"))


def test_deterministic_subset_ignores_file_order():
    rows = [{"row_id": i} for i in range(100)]
    a = train.deterministic_subset(rows, 10, seed=1)
    b = train.deterministic_subset(list(reversed(rows)), 10, seed=1)
    assert a == b and len(a) == 10


def test_targets_start_with_their_language_code(tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    rows = [{"row_id": 0, "english": "Hello there.", "din": "Dinka text.", "nuer": "Nuer text."}]
    ds = train.encode(tok, rows, ["eng_Latn-dik_Latn", "eng_Latn-nus_Latn"], 64, 64, seed=0)
    firsts = {item["labels"][0] for item in ds.items}
    assert firsts == {tok.convert_tokens_to_ids("dik_Latn"), tok.convert_tokens_to_ids("nus_Latn")}
    sources = {item["input_ids"][0] for item in ds.items}
    assert sources == {tok.convert_tokens_to_ids("eng_Latn")}


def test_unknown_direction_rejected(tmp_path, tiny_model):
    with pytest.raises(ValueError):
        train.main(["--model_name", str(tiny_model), "--model_revision", "", "--directions", "eng_Latn-fra_Latn"])


def test_training_run_writes_model_code_and_summary(tmp_path, tiny_model):
    train_dir = write_split(tmp_path / "train", 24)
    valid_dir = write_split(tmp_path / "valid", 6, offset=100)
    model_dir = tmp_path / "model"
    train.main([
        "--model_name", str(tiny_model), "--model_revision", "",
        "--train_rows", "16", "--valid_rows", "4", "--max_steps", "2", "--warmup_steps", "0",
        "--per_device_batch_size", "4", "--gradient_accumulation_steps", "1",
        "--eval_steps", "1", "--save_steps", "1", "--fp16", "false",
        "--train_dir", str(train_dir), "--valid_dir", str(valid_dir), "--model_dir", str(model_dir),
        "--checkpoint_dir", str(tmp_path / "ckpt"), "--output_data_dir", str(tmp_path / "out"),
    ])
    summary = json.loads((model_dir / "training_summary.json").read_text())
    assert summary["train_rows_used"] == 16 and summary["train_examples"] == 32
    assert summary["max_steps"] == 2 and summary["examples_seen"] == 8
    assert (model_dir / "config.json").exists()
    assert (model_dir / "code" / "inference.py").exists()
    assert (tmp_path / "out" / "training_summary.json").exists()


def test_serving_handler(tiny_model):
    loaded = inference.model_fn(str(tiny_model))
    out = inference.predict_fn({"inputs": ["Boil water before drinking."], "parameters": {"tgt_lang": "nus_Latn", "num_beams": 1, "max_new_tokens": 5}}, loaded)
    assert out["tgt_lang"] == "nus_Latn" and len(out["translations"]) == 1
    with pytest.raises(ValueError):
        inference.predict_fn({"inputs": "x", "parameters": {"tgt_lang": "fra_Latn"}}, loaded)
    with pytest.raises(ValueError):
        inference.predict_fn({"inputs": ["a"] * 17}, loaded)
