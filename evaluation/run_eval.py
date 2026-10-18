"""Translate the held-out test split with one model and score it with sacreBLEU.

Same script and same decoding settings for the base model and the fine-tuned
model. Writes metrics.json (BLEU and chrF per direction with the sacreBLEU
signatures, settings, test file hash) and predictions.jsonl.

The references are very probably machine generated (not confirmed by the
dataset author): the scores measure agreement with those references, not the
real quality for a speaker.
"""

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DIRECTIONS = {"eng_Latn-dik_Latn": "din", "eng_Latn-nus_Latn": "nuer"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_one(directory, pattern):
    files = sorted(Path(directory).glob(pattern))
    if len(files) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(files)}")
    return files[0]


def resolve_model(model):
    """A hub id, a model directory, or a directory holding SageMaker's model.tar.gz."""
    path = Path(model)
    if not path.exists():
        return model
    tars = list(path.glob("model.tar.gz")) if path.is_dir() else []
    if tars:
        target = path / "extracted"
        if not target.exists():
            with tarfile.open(tars[0]) as tar:
                tar.extractall(target, filter="data")
        return str(target)
    return str(path)


def translate(model, tokenizer, texts, src_lang, tgt_lang, num_beams, max_new_tokens, batch_size):
    """Batched beam search, longest inputs first, results returned in input order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts)
    tokenizer.src_lang = src_lang
    bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**batch, forced_bos_token_id=bos, num_beams=num_beams, max_new_tokens=max_new_tokens)
        for i, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="hub id, model directory, or directory with model.tar.gz")
    p.add_argument("--model_revision", default=None, help="hub revision (hub ids only)")
    p.add_argument("--label", required=True, help="name stored in the report, e.g. base or finetuned")
    p.add_argument("--test_dir", default="/opt/ml/processing/test")
    p.add_argument("--out_dir", default="/opt/ml/processing/evaluation")
    p.add_argument("--directions", default="eng_Latn-dik_Latn,eng_Latn-nus_Latn")
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="first N test rows only (smoke tests); 0 = all")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    test_file = find_one(args.test_dir, "*.jsonl")
    with open(test_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model(args.model)
    revision = args.model_revision if model_path == args.model else None
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, revision=revision, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    settings = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_source_tokens": 256,
        "dtype": "float16" if device == "cuda" else "float32",
    }
    report = {
        "label": args.label,
        "model": args.model if revision or not Path(args.model).exists() else "local-artifact",
        "model_revision": revision,
        "test_file_sha256": file_sha256(test_file),
        "test_rows": len(rows),
        "limited": bool(args.limit),
        "decoding": settings,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "directions": {},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    predictions = []
    for direction in [d for d in args.directions.split(",") if d]:
        src_lang, tgt_lang = direction.split("-")
        field = DIRECTIONS[direction]
        sources = [r["english"] for r in rows]
        refs = [r[field] for r in rows]
        start = time.time()
        hyps = translate(model, tokenizer, sources, src_lang, tgt_lang, args.num_beams, args.max_new_tokens, args.batch_size)
        seconds = time.time() - start
        bleu, chrf = BLEU(), CHRF()
        bleu_score = bleu.corpus_score(hyps, [refs])
        chrf_score = chrf.corpus_score(hyps, [refs])
        report["directions"][direction] = {
            "bleu": round(bleu_score.score, 2),
            "chrf": round(chrf_score.score, 2),
            "bleu_signature": str(bleu.get_signature()),
            "chrf_signature": str(chrf.get_signature()),
            "sentences": len(hyps),
            "seconds": round(seconds, 1),
        }
        print(f"{args.label} {direction}: BLEU {bleu_score.score:.2f} chrF {chrf_score.score:.2f} ({len(hyps)} sentences, {seconds:.0f} s)")
        for r, hyp in zip(rows, hyps):
            predictions.append({"row_id": r["row_id"], "direction": direction, "source": r["english"], "reference": r[field], "hypothesis": hyp})

    with open(Path(args.out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(Path(args.out_dir) / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
