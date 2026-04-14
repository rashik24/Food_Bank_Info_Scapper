import os
import json
import subprocess
from sklearn.model_selection import KFold
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from tqdm import tqdm
import time

start = time.time()
# conda activate py312
# export MKL_THREADING_LAYER=GNU
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python /home/rsiddiq2/test_CV_llama8.py
import time
# === Config ===
DATA_PATH = "/home/rsiddiq2/RUN/hours_new.jsonl"
DATA_DIR  = "/home/rsiddiq2/RUN"
BASE_OUT  = "/home/rsiddiq2/out-llama8-fold"
K = 5
CV_RESULTS = "/home/rsiddiq2/RUN/cv_results_8.json"

os.environ["MKL_THREADING_LAYER"] = "GNU"  # avoid MKL/OMP clash

# Ensure output dir exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

kf = KFold(n_splits=K, shuffle=True, random_state=42)

# === Prepare dataset_info.json entries ===
dataset_info = {}
for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
    train_file = f"fold{fold}_train.jsonl"
    val_file   = f"fold{fold}_val.jsonl"

    with open(f"{DATA_DIR}/{train_file}", "w", encoding="utf-8") as f:
        for rec in (data[i] for i in train_idx):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{DATA_DIR}/{val_file}", "w", encoding="utf-8") as f:
        for rec in (data[i] for i in val_idx):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for split, file in [("train", train_file), ("val", val_file)]:
        name = f"fold{fold}_{split}"
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

dataset_info_path = f"{DATA_DIR}/dataset_info.json"
with open(dataset_info_path, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)
print(f"\n✅ Wrote dataset_info.json at {dataset_info_path}")

# === Custom stopping criterion ===
class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_str):
        super().__init__()
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

def normalize_json(txt: str):
    try:
        obj = json.loads(txt)
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return txt.strip()

def make_window_tuples(data):
    if not isinstance(data, list):
        return set()
    return {
        (
            item.get("Day", "").lower(),
            item.get("Opening_Hour", "").strip().lower(),
            item.get("Closing_Hour", "").strip().lower(),
            tuple(item.get("Week", []))
        )
        for item in data
    }

def calculate_window_accuracy(model_data, reference_data):
    if not isinstance(reference_data, list) or not isinstance(model_data, list):
        return 0.0
    model_windows = make_window_tuples(model_data)
    ref_windows = make_window_tuples(reference_data)
    matches = ref_windows.intersection(model_windows)
    return (len(matches) / len(ref_windows) * 100) if ref_windows else 0.0

# === Cross-validation training + eval ===
all_metrics = []
for fold in range(1, K+1):
    out_dir = f"{BASE_OUT}{fold}"
    val_file = f"{DATA_DIR}/fold{fold}_val.jsonl"

    # --- Train ---
    cmd_train = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train",
        "--model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct",
        "--dataset", f"fold{fold}_train",
        "--dataset_dir", DATA_DIR,
        "--template", "llama3",
        "--finetuning_type", "lora",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "1",
        "--learning_rate", "2e-4",
        "--logging_steps", "1",
        "--val_size", "0",
        "--overwrite_output_dir",
        "--output_dir", out_dir,
        "--do_eval",
        "--eval_dataset", f"fold{fold}_val",
        "--eval_steps", "20",
        "--save_strategy", "steps",
        "--save_steps", "50",
        "--report_to", "none",
        "--disable_tqdm", "False"
    ]
    print(f"\n🚀 Training fold {fold} ...")
    subprocess.run(cmd_train, check=True)

    # --- Free memory from training ---
    torch.cuda.empty_cache()
    import gc
    

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # --- Evaluate in 8-bit ---
    print(f"🔎 Evaluating fold {fold} ...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(out_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model.to(device)

    results, window_accuracies = [], []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Fold {fold} eval"):
            rec = json.loads(line)
            user = rec["conversations"][0]["value"]
            gold = normalize_json(rec["conversations"][1]["value"])

            prompt = f"<|user|>\n{user}\n<|assistant|>"
            inputs = tok(prompt, return_tensors="pt").to(device)
            stop_criteria = StoppingCriteriaList([StopOnString(tok, "}]")])

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,  # reduce a bit for safety
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    stopping_criteria=stop_criteria
                )
            decoded = tok.decode(out[0], skip_special_tokens=True)
            if "<|assistant|>" in decoded:
                decoded = decoded.split("<|assistant|>")[-1].strip()
            if "]" in decoded:
                decoded = decoded[: decoded.rfind("]") + 1]
            try:
                json.loads(decoded)
                pred = normalize_json(decoded)
            except Exception:
                pred = decoded.strip()

            results.append({"input": user, "gold": gold, "prediction": pred})

            try:
                gold_obj, pred_obj = json.loads(gold), json.loads(pred)
                acc = calculate_window_accuracy(pred_obj, gold_obj)
                window_accuracies.append(acc)
            except Exception:
                continue

        mean_acc = sum(window_accuracies) / len(window_accuracies) if window_accuracies else 0.0
        metrics = {"fold": fold, "mean_window_accuracy": mean_acc, "samples": len(window_accuracies)}
        all_metrics.append(metrics)
        print(f"✅ Fold {fold} Mean Window Accuracy: {mean_acc:.2f}%")

        # --- Cleanup before next fold ---
        del model
        del tok
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# Save CV metrics
with open(CV_RESULTS, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
print(f"\n✅ Cross-validation finished. Results saved in {CV_RESULTS}")
import time
print(f"Runtime: {time.time() - start:.4f} seconds")