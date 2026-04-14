#!/usr/bin/env python3
# hp_train_final.py

import os
import json
import argparse
import subprocess
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

# =====================================================
# CLI
# =====================================================
parser = argparse.ArgumentParser(description="Train final LoRA model on full data")
parser.add_argument("--data", type=str, required=True,
                    help="Path to full JSONL training dataset (sharegpt format)")
parser.add_argument("--config", type=str, required=True,
                    help="Path to best_config.json")
parser.add_argument("--out", type=str, required=True,
                    help="Output directory for final trained model")
parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                    help="Base model to fine-tune")
parser.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs (default 3)")
args = parser.parse_args()

# =====================================================
# LOAD BEST CONFIG
# =====================================================
if not os.path.exists(args.config):
    raise FileNotFoundError(f"❌ Missing config file: {args.config}")

with open(args.config, "r", encoding="utf-8") as f:
    cfg = json.load(f)

lora_r = cfg.get("lora_r", 8)
lora_alpha = cfg.get("lora_alpha", 32)
lr = cfg.get("lr", "2e-4")

print(f"🧠 Training with config: r={lora_r}, α={lora_alpha}, lr={lr}")

# =====================================================
# REGISTER DATASET FOR LLAMA FACTORY
# =====================================================
data_dir = os.path.dirname(args.data)
dataset_info_path = os.path.join(data_dir, "dataset_info.json")

dataset_info = {
    "full_train": {
        "path": data_dir,
        "file_name": os.path.basename(args.data),
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": None},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}

with open(dataset_info_path, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)
print(f"📘 Dataset registered in {dataset_info_path}")

# =====================================================
# TRAIN COMMAND
# =====================================================
os.makedirs(args.out, exist_ok=True)

cmd_train = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",
    "--model_name_or_path", args.base_model,
    "--dataset", "full_train",
    "--dataset_dir", data_dir,
    "--template", "llama3",
    "--finetuning_type", "lora",
    "--lora_rank", str(lora_r),
    "--lora_alpha", str(lora_alpha),
    "--num_train_epochs", str(args.epochs),
    "--per_device_train_batch_size", "1",
    "--learning_rate", str(lr),
    "--cutoff_len", "512",
    "--overwrite_output_dir",
    "--output_dir", args.out,
    "--save_strategy", "epoch",
    "--report_to", "none",
    "--disable_tqdm", "False"
]

print(f"\n🚀 Starting final training...")
subprocess.run(cmd_train, check=True)
print(f"✅ Final model saved to {args.out}")



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json, gc, os

# =====================================================
# STEP: Inference on held-out test data
# =====================================================
print("🔎 Starting inference on test set...")

test_file = os.path.join(os.path.dirname(args.data), f"test_subset_{os.path.basename(args.data).split('_')[-1]}")
if not os.path.exists(test_file):
    print(f"⚠️ Test file not found at {test_file}. Skipping inference.")
    exit(0)

# Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(args.out, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.out,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

def run_model(prompt: str) -> str:
    formatted = f"<|user|>\n{prompt.strip()}\n<|assistant|>"
    inputs = tok(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()
    return decoded

pred_file = os.path.join(args.out, "pred_gold.jsonl")

with open(test_file, "r", encoding="utf-8") as fin, open(pred_file, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Generating predictions"):
        rec = json.loads(line)
        user_msg = rec["conversations"][0]["value"]
        gold_text = rec["conversations"][1]["value"]
        pred_text = run_model(user_msg)
        fout.write(json.dumps({"prompt_text": user_msg,  "gold_text": gold_text, "pred_text": pred_text}, ensure_ascii=False) + "\n")

print(f"✅ Predictions saved to {pred_file}")

# Clean up GPU memory
del model, tok
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



