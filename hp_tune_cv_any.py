#!/usr/bin/env python3
# hp_tune_cv_any.py
import os
import json
import subprocess
import argparse
from sklearn.model_selection import KFold

# =====================================================
# CLI ARGUMENTS
# =====================================================
parser = argparse.ArgumentParser(description="Cross-validated hyperparameter tuning driver.")
parser.add_argument("--train_size", type=int, required=True,
                    help="Number of samples to use for HP tuning (e.g., 150, 200, 250)")
parser.add_argument("--data_file", type=str, default="/home/rsiddiq2/master_run/hours_all_merged.jsonl",
                    help="Full dataset JSONL path")
parser.add_argument("--out_dir", type=str, default="/home/rsiddiq2/master_run",
                    help="Base output directory for all folds/configs")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--worker", type=str, default="hp_worker.py",
                    help="Worker script to run LoRA fine-tuning/inference per fold.")
args = parser.parse_args()

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = str(args.out_dir)
FULL_DATA = args.data_file
TRAIN_SIZE = args.train_size
OUT_SUMMARY = os.path.join(DATA_DIR, f"hp_tune_cv{TRAIN_SIZE}_summary.json")

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

K = 5  # number of CV folds
os.makedirs(DATA_DIR, exist_ok=True)

# LoRA hyperparameter grid
HP_GRID = [
    (4, 16, "2e-4"),
   # (8, 32, "1e-4"),
    #(16, 32, "1e-4"),
 # (16, 64, "5e-5"),
   # (8, 16, "2e-4")
]

# =====================================================
# LOAD DATA
# =====================================================
with open(FULL_DATA, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

train_data = all_data[:TRAIN_SIZE]
print(f"✅ Loaded {len(train_data)} samples for {K}-fold CV.")

# =====================================================
# HELPER
# =====================================================
def write_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# =====================================================
# MAIN LOOP
# =====================================================
kf = KFold(n_splits=K, shuffle=True, random_state=42)
run_log = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data), 1):
    print(f"\n==============================")
    print(f"🌀 Fold {fold}/{K} for CV{TRAIN_SIZE}")
    print("==============================")

    # Prepare fold data
    fold_dir = os.path.join(DATA_DIR, f"out_hp_{TRAIN_SIZE}")
    os.makedirs(fold_dir, exist_ok=True)

    train_file = os.path.join(fold_dir, f"cv{TRAIN_SIZE}_fold{fold}_train.jsonl")
    val_file   = os.path.join(fold_dir, f"cv{TRAIN_SIZE}_fold{fold}_val.jsonl")
    write_jsonl([train_data[i] for i in train_idx], train_file)
    write_jsonl([train_data[i] for i in val_idx], val_file)

    # Run LoRA worker for each config
    for (r, a, lr) in HP_GRID:
        tag = f"cv{TRAIN_SIZE}_fold{fold}_r{r}_a{a}_lr{lr.replace('.', '')}"
        print(f"\n🚀 Running {tag}")

        # Folder per fold-config combination
        out_dir = os.path.join(fold_dir, f"cv{TRAIN_SIZE}_r{r}_a{a}_fold{fold}")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", args.worker,
            "--train_file", train_file,
            "--val_file", val_file,
            "--fold", str(fold),
            "--lora_r", str(r),
            "--lora_alpha", str(a),
            "--lr", lr,
            "--epochs", str(args.epochs)
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed {tag}: {e}")
            continue

        pred_file = os.path.join(out_dir, "pred_gold.jsonl")
        run_log.append({
            "cv_tag": tag,
            "fold": fold,
            "train_size": TRAIN_SIZE,
            "lora_r": r,
            "lora_alpha": a,
            "lr": lr,
            "train_file": train_file,
            "val_file": val_file,
            "pred_gold_file": pred_file
        })

# =====================================================
# SAVE SUMMARY
# =====================================================
with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(run_log, f, indent=2, ensure_ascii=False)

print("\n✅ Cross-validated HP tuning completed.")
print(f"📦 Summary saved to {OUT_SUMMARY}")
print("Each run’s predictions are stored in their respective out_hp_*_fold*/ folders.")


# =====================================================
# 🔍 DIAGNOSTIC: Check fold uniqueness
# =====================================================
import hashlib

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print("\n=== 🔍 DATA & OUTPUT SANITY CHECK ===")

train_hashes, val_hashes, pred_hashes = [], [], []
for fold in range(1, K+1):
    tfile = os.path.join(DATA_DIR, f"out_hp_{TRAIN_SIZE}", f"cv{TRAIN_SIZE}_fold{fold}_train.jsonl")
    vfile = os.path.join(DATA_DIR, f"out_hp_{TRAIN_SIZE}", f"cv{TRAIN_SIZE}_fold{fold}_val.jsonl")

    pfile = os.path.join(DATA_DIR, f"out_hp_{TRAIN_SIZE}", f"cv{TRAIN_SIZE}_r{HP_GRID[0][0]}_a{HP_GRID[0][1]}_fold{fold}", "pred_gold.jsonl")

    for ftype, path in [("train", tfile), ("val", vfile), ("pred", pfile)]:
        if os.path.exists(path):
            h = hash_file(path)
            (train_hashes if ftype=="train" else val_hashes if ftype=="val" else pred_hashes).append((fold, h))
        else:
            print(f"⚠️ Missing {ftype} file for fold {fold}: {path}")

print("\nTrain hashes:")
for f,h in train_hashes: print(f"  Fold {f}: {h[:10]}...")
print("\nVal hashes:")
for f,h in val_hashes: print(f"  Fold {f}: {h[:10]}...")
print("\nPred hashes:")
for f,h in pred_hashes: print(f"  Fold {f}: {h[:10]}...")

if len({h for _,h in train_hashes}) == 1:
    print("⚠️ All train folds identical — data split issue.")
if len({h for _,h in val_hashes}) == 1:
    print("⚠️ All val folds identical — data leakage.")
if len({h for _,h in pred_hashes}) == 1:
    print("⚠️ All predictions identical — model reuse or bad fold separation.")
else:
    print("✅ Folds differ — CV working correctly.")




