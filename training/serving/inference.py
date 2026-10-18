"""Serving handlers for the SageMaker Hugging Face inference toolkit.

Request:  {"inputs": "text" or ["text", ...],
           "parameters": {"tgt_lang": "dik_Latn" | "nus_Latn", "num_beams": 4, "max_new_tokens": 200}}
Response: {"translations": [...], "tgt_lang": ..., "input_words": n, "output_words": n}

Only English input is supported (src_lang eng_Latn), the direction the model
was fine-tuned on.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SRC_LANG = "eng_Latn"
TARGETS = ("dik_Latn", "nus_Latn")
MAX_INPUTS = 16
MAX_INPUT_CHARS = 1000


def model_fn(model_dir, context=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    return model, tokenizer


def predict_fn(data, model_and_tokenizer, context=None):
    model, tokenizer = model_and_tokenizer
    inputs = data.get("inputs")
    texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
    if not texts or len(texts) > MAX_INPUTS or any(not isinstance(t, str) or not t.strip() or len(t) > MAX_INPUT_CHARS for t in texts):
        raise ValueError(f"inputs must be 1 to {MAX_INPUTS} non-empty strings of at most {MAX_INPUT_CHARS} characters")
    params = data.get("parameters") or {}
    tgt_lang = params.get("tgt_lang", "dik_Latn")
    if tgt_lang not in TARGETS:
        raise ValueError(f"tgt_lang must be one of {TARGETS}")
    num_beams = int(params.get("num_beams", 4))
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 256)

    tokenizer.src_lang = SRC_LANG
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    translations = tokenizer.batch_decode(out, skip_special_tokens=True)
    return {
        "translations": translations,
        "tgt_lang": tgt_lang,
        "input_words": sum(len(t.split()) for t in texts),
        "output_words": sum(len(t.split()) for t in translations),
    }
