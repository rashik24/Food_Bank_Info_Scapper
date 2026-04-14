#!/usr/bin/env python3
# hp_eval_results.py
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


# =====================================================
# CLI
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="Path to pred_gold.jsonl file")
parser.add_argument("--out_csv", type=str, required=True,
                    help="Path to save evaluation CSV summary")
parser.add_argument("--hours_parser", type=str, default="/home/rsiddiq2/LLaMA-Factory/out-llama3/",
                    help="Fine-tuned JSON parser model directory")
args = parser.parse_args()

PRED_FILE = args.input
OUT_CSV = args.out_csv
HOURS_PARSER = args.hours_parser


# =====================================================
# LOAD PARSER MODEL
# =====================================================
print("🧠 Loading JSON parser model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained(HOURS_PARSER, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    HOURS_PARSER,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
).eval()


# =====================================================
# HELPERS
# =====================================================
class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_str):
        super().__init__()
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids


def run_model(text: str) -> str:
    if not text.strip():
        return "[]"
    prompt = f"<|user|>\n{text.strip()}\n<|assistant|>"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    stop_criteria = StoppingCriteriaList([StopOnString(tok, "}]")])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=stop_criteria,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()
    if "]" in decoded:
        decoded = decoded[: decoded.rfind("]") + 1]
    return decoded.strip()


def normalize_with_llm(raw_text: str):
    if not raw_text.strip():
        return []
    parsed_text = run_model(raw_text)
    try:
        parsed_text = parsed_text.strip()
        if parsed_text.startswith("[") and parsed_text.endswith("]"):
            json_str = parsed_text
        elif "[" in parsed_text and "]" in parsed_text:
            json_str = parsed_text[parsed_text.index("["): parsed_text.rindex("]") + 1]
        elif "{" in parsed_text and "}" in parsed_text:
            json_str = parsed_text[parsed_text.index("{"): parsed_text.rindex("}") + 1]
        else:
            return []
        parsed = json.loads(json_str)
        return [parsed] if isinstance(parsed, dict) else parsed
    except Exception:
        return []

import math

def make_window_tuples(data):
    """
    Convert list of dicts into atomic
    (day, open, close, week) tuples.
    Skips NaN, None, strings, and malformed items.
    """
    atoms = set()

    if not isinstance(data, list):
        return atoms

    for item in data:

        # Skip NaN
        if isinstance(item, float) and math.isnan(item):
            continue

        # Skip None
        if item is None:
            continue

        # Skip non-dict (str, int, etc.)
        if not isinstance(item, dict):
            continue

        day = str(item.get("Day", "")).strip().lower()
        open_h = str(item.get("Opening_Hour", "")).strip().lower()
        close_h = str(item.get("Closing_Hour", "")).strip().lower()

        weeks = item.get("Week", [])

        if weeks is None:
            continue

        if not isinstance(weeks, list):
            weeks = [weeks]

        for w in weeks:
            if w is None:
                continue
            atoms.add((day, open_h, close_h, str(w)))

    return atoms


def calculate_window_accuracy(model_data, reference_data):

    model_windows = make_window_tuples(model_data)
    ref_windows   = make_window_tuples(reference_data)

    # ✅ BOTH missing → perfect
    if not model_windows and not ref_windows:
        return 1.0

    # ✅ Gold exists but model empty → total failure
    if ref_windows and not model_windows:
        return 0.0

    # ✅ No gold (shouldn't happen often)
    if not ref_windows:
        return 1.0

    matches = model_windows & ref_windows
    return len(matches) / len(ref_windows)




# def calculate_window_accuracy(gold_list, pred_list):
#     if not isinstance(gold_list, list) or not isinstance(pred_list, list):
#         return 0.0
#     if not gold_list and not pred_list:
#         return 1.0
#     if not gold_list or not pred_list:
#         return 0.0
#     gold_valid = [g for g in gold_list if isinstance(g, dict)]
#     pred_valid = [p for p in pred_list if isinstance(p, dict)]
#     if not gold_valid or not pred_valid:
#         return 0.0
#     matches = 0
#     gold_copy = gold_valid.copy()
#     for p in pred_valid:
#         for g in gold_copy:
#             if (
#                 str(p.get("Day")) == str(g.get("Day")) and
#                 str(p.get("Opening_Hour")) == str(g.get("Opening_Hour")) and
#                 str(p.get("Closing_Hour")) == str(g.get("Closing_Hour"))
#             ):
#                 matches += 1
#                 gold_copy.remove(g)
#                 break
#     denom = max(len(gold_valid), len(pred_valid))
#     return matches / denom if denom > 0 else 0.0


# =====================================================
# MAIN
# =====================================================
print(f"📄 Reading predictions: {PRED_FILE}")
with open(PRED_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

records = []
for row in tqdm(data, desc="Evaluating results"):
    gold = str(row.get("gold_text", "")).strip()
    pred = str(row.get("pred_text", "")).strip()
    if not gold and not pred:
        continue
    gold_json = normalize_with_llm(gold)
    pred_json = normalize_with_llm(pred)
    win_acc = calculate_window_accuracy(gold_json, pred_json)
    records.append({"mean_window_accuracy": win_acc})

df = pd.DataFrame(records)
mean_wa = df["mean_window_accuracy"].mean() if not df.empty else 0.0

print(f"\n✅ Mean window accuracy: {mean_wa:.4f}")
df.to_csv(OUT_CSV, index=False)
print(f"💾 Saved → {OUT_CSV}")



