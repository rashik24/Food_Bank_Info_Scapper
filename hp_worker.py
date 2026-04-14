# worker_infer_hp.py
import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from tqdm import tqdm

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", required=True)
parser.add_argument("--val_file", required=True)
parser.add_argument("--fold", required=True)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lr", default="2e-4")
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

DATA_DIR = os.path.dirname(args.train_file)
train_name = f"train_fold{args.fold}"
val_name   = f"val_fold{args.fold}"
# use consistent structure under the same base dir
base_dir = os.path.dirname(args.train_file)          # same directory where CV data files live
out_dir = os.path.join(
    base_dir,
    f"cv{os.path.basename(base_dir).split('_')[-1]}_r{args.lora_r}_a{args.lora_alpha}_fold{args.fold}"
)
os.makedirs(out_dir, exist_ok=True)

os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Register dataset for LLaMA Factory
# -----------------------------
dataset_info_path = os.path.join(DATA_DIR, "dataset_info.json")
if os.path.exists(dataset_info_path):
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}

for name, file in [(train_name, os.path.basename(args.train_file)),
                   (val_name, os.path.basename(args.val_file))]:
    dataset_info[name] = {
        "path": DATA_DIR,
        "file_name": file,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": None},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }

with open(dataset_info_path, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

# -----------------------------
# Train (no eval)
# -----------------------------
cmd_train = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",
    "--model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct",
    "--dataset", train_name,
    "--dataset_dir", DATA_DIR,
    "--template", "llama3",
    "--finetuning_type", "lora",
    "--lora_rank", str(args.lora_r),
    "--lora_alpha", str(args.lora_alpha),
    "--num_train_epochs", str(args.epochs),
    "--per_device_train_batch_size", "1",
    "--learning_rate", args.lr,
    "--cutoff_len", "512",
    "--overwrite_output_dir",
    "--output_dir", out_dir,
    "--save_strategy", "epoch",
    "--report_to", "none",
    "--disable_tqdm", "False"
]
os.system(" ".join(cmd_train))

# -----------------------------
# Load fine-tuned model for inference
# -----------------------------
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(out_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    out_dir,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Inference loop (no eval)
# -----------------------------
def run_model(prompt: str) -> str:
    formatted = f"<|user|>\n{prompt}\n<|assistant|>"
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

# store only predictions + gold text
out_jsonl = os.path.join(out_dir, "pred_gold.jsonl")

with open(args.val_file, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc=f"Fold {args.fold} inference"):
        rec = json.loads(line)
        user_msg = rec["conversations"][0]["value"]
        gold_text = rec["conversations"][1]["value"]
        pred_text = run_model(user_msg)
        fout.write(json.dumps({"prompt_text": user_msg,"pred_text": pred_text, "gold_text": gold_text}, ensure_ascii=False) + "\n")

print(f"✅ Fold {args.fold} done. Saved to {out_jsonl}")

# -----------------------------
# Cleanup
# -----------------------------
del model, tok
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
