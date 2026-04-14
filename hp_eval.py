#!/usr/bin/env python3
# hp_eval.py
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
# CLI ARGS
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="Path to hp_tune_cv*_summary.json")
parser.add_argument("--out_csv", type=str, required=True,
                    help="Path to save evaluation CSV summary")
parser.add_argument("--output", type=str, default=None,
                    help="Optional path for best_config.json")
parser.add_argument("--hours_parser", type=str, default="/home/rsiddiq2/LLaMA-Factory/out-llama3/",
                    help="Fine-tuned JSON parser directory")
args = parser.parse_args()


SUMMARY_FILE = args.input
OUT_CSV = args.out_csv
BEST_CONFIG_PATH = args.output or os.path.join(os.path.dirname(OUT_CSV), "best_config.json")
HOURS_PARSER = args.hours_parser


os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)


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
    local_files_only=True,
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
    if not text or not text.strip():
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
    if not raw_text or not raw_text.strip():
        return []
    parsed_text = run_model(raw_text)
    try:
        parsed_text = parsed_text.strip()
        if parsed_text.startswith("[") and parsed_text.endswith("]"):
            json_str = parsed_text
        elif "[" in parsed_text and "]" in parsed_text:
            json_str = parsed_text[parsed_text.index("[") : parsed_text.rindex("]") + 1]
        elif "{" in parsed_text and "}" in parsed_text:
            json_str = parsed_text[parsed_text.index("{") : parsed_text.rindex("}") + 1]
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
#     total_fields = 4 * len(gold_valid)  # 4 fields per window
#     gold_copy = gold_valid.copy()

#     for p in pred_valid:
#         for g in gold_copy:
#             score = 0
#             if str(p.get("Day")) == str(g.get("Day")):
#                 score += 1
#             if str(p.get("Opening_Hour")) == str(g.get("Opening_Hour")):
#                 score += 1
#             if str(p.get("Closing_Hour")) == str(g.get("Closing_Hour")):
#                 score += 1
#             if str(p.get("Week")) == str(g.get("Week")):
#                 score += 1

#             if score == 4:  # All 4 elements match perfectly
#                 matches += 4
#                 gold_copy.remove(g)
#                 break
#             else:
#                 matches += score  # partial match contributes proportionally

#     return (matches / total_fields) * 100 if total_fields > 0 else 0.0



# =====================================================
# MAIN
# =====================================================
print(f"📄 Reading summary file: {SUMMARY_FILE}")
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
    f.seek(0)
    if first_line.startswith("{") or first_line.startswith('{"'):
        # multiple lines of JSON objects
        run_log = [json.loads(line) for line in f if line.strip()]
    else:
        # single JSON object or list
        run_log = json.load(f)



if not run_log:
    print("⚠️ No runs found in summary file.")
    exit(0)


records = []
for run in tqdm(run_log, desc="Evaluating HP runs"):
    pred_path = run["pred_gold_file"]
    if not os.path.exists(pred_path):
        print(f"⚠️ Missing predictions file: {pred_path}")
        continue


    with open(pred_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    #data = data[:2]


    win_accs = []
    for row in data:
        print(row)
        gold = str(row.get("gold_text", "")).strip()
        pred = str(row.get("pred_text", "")).strip()
        if not gold and not pred:
            continue
        gold_json = normalize_with_llm(gold)
        pred_json = normalize_with_llm(pred)
        win_acc = calculate_window_accuracy(gold_json, pred_json)
        win_accs.append(win_acc)


    mean_wa = sum(win_accs) / len(win_accs) if win_accs else 0.0
    records.append({
        "cv_tag": run["cv_tag"],
        "train_size": run["train_size"],
        "fold": run["fold"],
        "lora_r": run["lora_r"],
        "lora_alpha": run["lora_alpha"],
        "lr": run["lr"],
        "mean_window_accuracy": round(mean_wa, 4),
        "samples": len(win_accs),
    })


df = pd.DataFrame(records)
if df.empty:
    print("⚠️ No valid results to evaluate.")
    exit(0)


df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved summary to {OUT_CSV}")
print(df)


# =====================================================
# Select best config
# =====================================================
best_row = df.loc[df["mean_window_accuracy"].idxmax()]
best_config = {
    "lora_r": int(best_row["lora_r"]),
    "lora_alpha": int(best_row["lora_alpha"]),
    "lr": best_row["lr"],
}
with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(best_config, f, indent=2)
print(f"\n🏆 Best config: {best_config}")
print(f"💾 Saved → {BEST_CONFIG_PATH}")





