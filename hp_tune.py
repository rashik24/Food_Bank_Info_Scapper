#!/usr/bin/env python3
# hp_tune_cv200.py
import os
import json
import subprocess
from sklearn.model_selection import KFold

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "/home/rsiddiq2/master_run"
FULL_DATA = os.path.join(DATA_DIR, "hours_all_merged.jsonl")
OUT_SUMMARY = os.path.join(DATA_DIR, "hp_tune_cv200_summary.json")
os.environ["MKL_THREADING_LAYER"] = "INTEL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# 5-fold CV settings
K = 5
os.makedirs(DATA_DIR, exist_ok=True)

# LoRA hyperparameter grid
HP_GRID = [
    (4, 16, "2e-4"),
    (8, 16, "2e-4"),
    (8, 32, "1e-4"),
    (16, 32, "1e-4"),
    (16, 64, "5e-5"),
]

# =====================================================
# LOAD DATA
# =====================================================
with open(FULL_DATA, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

train200 = all_data[:200]   # first 200 samples for CV
print(f"Loaded {len(train200)} samples for 5-fold CV.")

# =====================================================
# HELPER: write JSONL
# =====================================================
def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# =====================================================
# MAIN LOOP
# =====================================================
kf = KFold(n_splits=K, shuffle=True, random_state=42)
run_log = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train200), 1):
    print(f"\n==============================")
    print(f"📘 Fold {fold}/{K}")
    print("==============================")

    train_file = os.path.join(DATA_DIR, f"cv200_fold{fold}_train.jsonl")
    val_file   = os.path.join(DATA_DIR, f"cv200_fold{fold}_val.jsonl")

    write_jsonl([train200[i] for i in train_idx], train_file)
    write_jsonl([train200[i] for i in val_idx], val_file)

    # Loop through hyperparameter configs
    for (r, a, lr) in HP_GRID:
        tag = f"fold{fold}_r{r}_a{a}_lr{lr.replace('.', '')}"
        print(f"\n🚀 Running {tag}")

        cmd = [
            "python", "worker_hp.py",
            "--train_file", train_file,
            "--val_file", val_file,
            "--fold", str(fold),
            "--lora_r", str(r),
            "--lora_alpha", str(a),
            "--lr", lr,
            "--epochs", "3"
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed {tag}: {e}")
            continue

        out_dir = f"/home/rsiddiq2/out_hp_r{r}_a{a}_fold{fold}"
        pred_file = os.path.join(out_dir, "pred_gold.jsonl")

        run_log.append({
            "fold": fold,
            "lora_r": r,
            "lora_alpha": a,
            "lr": lr,
            "train_file": train_file,
            "val_file": val_file,
            "output_dir": out_dir,
            "pred_gold_file": pred_file
        })

# =====================================================
# SAVE SUMMARY
# =====================================================
with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(run_log, f, indent=2, ensure_ascii=False)

print("\n✅ Cross-validated HP tuning completed.")
print(f"🗂 Summary saved to {OUT_SUMMARY}")
print("Each run’s predictions are in its out_hp_r*_a*_fold*/pred_gold.jsonl folder.")


