"""Fine-tune NLLB-200 on English-Dinka and English-Nuer pairs (SageMaker training job).

Runs in the Hugging Face PyTorch training container. Reads JSON Lines splits
from the SageMaker channels, trains on a deterministic subset of the training
split for a capped number of steps, reports validation loss, and writes the
model, the tokenizer, the serving code and a training summary to the model
directory. The test split is never read here.

Dataset: English-Nuer-Dinka Parallel Corpus by Tajnaam (CC BY-NC 4.0).
Base model: facebook/nllb-200-distilled-600M (CC BY-NC 4.0).
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Direction -> corpus field holding the target text.
DIRECTIONS = {"eng_Latn-dik_Latn": "din", "eng_Latn-nus_Latn": "nuer"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_jsonl(directory):
    files = sorted(Path(directory).glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one .jsonl file in {directory}, found {len(files)}")
    return files[0]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_subset(rows, n, seed):
    """First n rows in SHA-256 order of seed:row_id (independent of file order)."""
    if n <= 0 or n >= len(rows):
        return list(rows)
    key = lambda r: hashlib.sha256(f"{seed}:{r['row_id']}".encode()).hexdigest()
    return sorted(rows, key=key)[:n]


class Features(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def encode(tokenizer, rows, directions, max_source_length, max_target_length, seed):
    """Tokenise every row once per direction; the target starts with its language code."""
    items = []
    for direction in directions:
        src_lang, tgt_lang = direction.split("-")
        field = DIRECTIONS[direction]
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        enc = tokenizer(
            [r["english"] for r in rows],
            text_target=[r[field] for r in rows],
            max_length=max_source_length,
            truncation=True,
        )
        for i in range(len(rows)):
            labels = enc["labels"][i][:max_target_length]
            items.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": labels})
    random.Random(seed).shuffle(items)
    return Features(items)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_name", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--model_revision", default="f8d333a098d19b4fd9a8b18f94170487ad3f821d")
    p.add_argument("--directions", default="eng_Latn-dik_Latn,eng_Latn-nus_Latn")
    p.add_argument("--train_rows", type=int, default=50000, help="rows of the training split used (each gives one example per direction)")
    p.add_argument("--valid_rows", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--per_device_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_source_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--fp16", type=lambda s: s.lower() == "true", default=torch.cuda.is_available())
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--stress_max_length", type=lambda s: s.lower() == "true", default=False,
                   help="memory check: longest training rows, every batch padded to the maximum length")
    p.add_argument("--train_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"))
    p.add_argument("--valid_dir", default=os.environ.get("SM_CHANNEL_VALID", "data/processed/valid"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "outputs/model"))
    p.add_argument("--checkpoint_dir", default="/opt/ml/checkpoints" if os.path.isdir("/opt/ml") else "outputs/checkpoints")
    p.add_argument("--output_data_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "outputs/data"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    directions = [d for d in args.directions.split(",") if d]
    unknown = set(directions) - set(DIRECTIONS)
    if unknown:
        raise ValueError(f"unknown directions: {sorted(unknown)}")

    train_file, valid_file = find_jsonl(args.train_dir), find_jsonl(args.valid_dir)
    train_all = load_jsonl(train_file)
    if args.stress_max_length:
        longest = lambda r: len(r["english"]) + max(len(r["din"]), len(r["nuer"]))
        train_rows = sorted(train_all, key=longest, reverse=True)[: args.train_rows]
    else:
        train_rows = deterministic_subset(train_all, args.train_rows, args.seed)
    valid_rows = deterministic_subset(load_jsonl(valid_file), args.valid_rows, args.seed)
    print(f"train rows used {len(train_rows)} of {len(train_all)}, valid rows {len(valid_rows)}, directions {directions}")
    del train_all

    revision = args.model_revision or None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, revision=revision)
    train_ds = encode(tokenizer, train_rows, directions, args.max_source_length, args.max_target_length, args.seed)
    valid_ds = encode(tokenizer, valid_rows, directions, args.max_source_length, args.max_target_length, args.seed)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim="adafactor",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=max(1, min(50, args.max_steps // 10)),
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        group_by_length=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=max(args.max_source_length, args.max_target_length) if args.stress_max_length else None,
        ),
        processing_class=tokenizer,
    )

    start = time.time()
    eval_before = trainer.evaluate()
    print(f"eval_loss_before={eval_before['eval_loss']:.4f}")
    last = get_last_checkpoint(args.checkpoint_dir) if os.path.isdir(args.checkpoint_dir) else None
    if last:
        print(f"resuming from checkpoint {last}")
    result = trainer.train(resume_from_checkpoint=last)
    eval_after = trainer.evaluate()
    print(f"eval_loss_after={eval_after['eval_loss']:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    serving = Path(__file__).resolve().parent / "serving"
    if serving.is_dir():
        shutil.copytree(serving, Path(args.model_dir) / "code", dirs_exist_ok=True)

    summary = {
        "base_model": args.model_name,
        "base_model_revision": args.model_revision,
        "directions": directions,
        "train_file_sha256": file_sha256(train_file),
        "valid_file_sha256": file_sha256(valid_file),
        "train_rows_used": len(train_rows),
        "train_examples": len(train_ds),
        "valid_rows_used": len(valid_rows),
        "max_steps": args.max_steps,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "examples_seen": result.global_step * args.per_device_batch_size * args.gradient_accumulation_steps,
        "epochs_over_subset": round(result.global_step * args.per_device_batch_size * args.gradient_accumulation_steps / max(1, len(train_ds)), 3),
        "learning_rate": args.learning_rate,
        "optimizer": "adafactor",
        "fp16": args.fp16,
        "seed": args.seed,
        "train_loss": result.training_loss,
        "eval_loss_before": eval_before["eval_loss"],
        "eval_loss_after": eval_after["eval_loss"],
        "train_seconds": round(time.time() - start, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "stress_max_length": args.stress_max_length,
        "peak_gpu_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None,
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
    }
    os.makedirs(args.output_data_dir, exist_ok=True)
    for target in (Path(args.model_dir) / "training_summary.json", Path(args.output_data_dir) / "training_summary.json"):
        target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
