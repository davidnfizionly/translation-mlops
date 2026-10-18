"""Tiny randomly initialised model with the NLLB-200 architecture and the real NLLB tokenizer.

Used by the offline tests: same code paths as the real model, seconds on a CPU.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer, M2M100Config, M2M100ForConditionalGeneration

BASE = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def build_tiny_model(directory):
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    config = M2M100Config(
        vocab_size=len(tokenizer),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
    model = M2M100ForConditionalGeneration(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return Path(directory)


def write_split(directory, n, offset=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "split.jsonl", "w", encoding="utf-8") as f:
        for i in range(offset, offset + n):
            row = {"row_id": i, "english": f"The village number {i} has a school.", "nuer": f"Nuer sentence {i} ɛ wec.", "din": f"Dinka sentence {i} ë baai."}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return directory
