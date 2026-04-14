import os
import json
import random
import subprocess
import datetime
import pandas as pd
from pathlib import Path
import time 
start = time.time()

#hours_hil_text_old_method_test.jsonl
# ============================================================
# CONFIGURATION
# ============================================================

#DATA_FILE = "/home/rsiddiq2/Run_3/hours_hil_text_old_method_test.jsonl"    
DATA_FILE = "/home/rsiddiq2/Run_3/hours_hil_text_All_Jan.jsonl"    
#/home/rsiddiq2/Run_3/             # dataset inside master_run/
RUN_STEPS = [100]            # progressive training sizes
TEST_SIZE = 50                         # next chunk used as test
SEED = 100                             # reproducibility
HP_SCRIPT = "hp_command.py"            # script for hyperparameter CV
TRAIN_SCRIPT = "hp_train.py"              # script for full training
PARSER_SCRIPT = "hp_eval.py"       # script for JSON parsing
RESULTS_CSV = "master_results_new.csv"     # cumulative results
LOG_DIR = "logs"
BASE_DIR = Path("/home/rsiddiq2/Run_3")
# ============================================================
# SETUP
# ============================================================

Path(LOG_DIR).mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOG_DIR, f"run_{timestamp}.log")

def log(msg):
    """Log both to console and file"""
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()}: {msg}\n")

random.seed(SEED)

# ============================================================
# LOAD & SHUFFLE DATA
# ============================================================

log("🔄 Loading and shuffling data...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
random.seed(42) 
random.shuffle(data)
total_records = len(data)
log(f"✅ Loaded {total_records} records.")

# ============================================================
# MAIN LOOP
# ============================================================

results = []

for train_size in RUN_STEPS:
    log(f"\n🚀 Starting run for TRAIN_SIZE={train_size}...")

    # Prepare subsets
    train_subset = data[:train_size]
    test_subset  = data[train_size:train_size+TEST_SIZE]

    train_file = f"train_subset_{train_size}.jsonl"
    test_file  = f"test_subset_{train_size}.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_subset:
            f.write(json.dumps(row) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for row in test_subset:
            f.write(json.dumps(row) + "\n")

    out_hp_dir = BASE_DIR / f"out_hp_{train_size}"
    out_train_dir = BASE_DIR / f"out_train_{train_size}"
    Path(out_hp_dir).mkdir(exist_ok=True)
    Path(out_train_dir).mkdir(exist_ok=True)

    # ========================================================
    # STEP 1 — Hyperparameter tuning with CV
    # ========================================================
    log(f"🎯 Running HP tuning on {train_size} samples...")
    subprocess.run(
    [
        "python", "hp_tune_cv_any.py",
        "--train_size", str(train_size),
        "--data_file", str(DATA_FILE),
        "--out_dir", out_hp_dir
    ],
    check=True
)


    # ========================================================
    # STEP 2 — Parse best config from HP results
    # ========================================================
    log("🧩 Extracting best config...")


    # 1️⃣ Run the JSON parser to evaluate all pred_gold.jsonl files
    hp_eval_csv = os.path.join(out_hp_dir, f"hp_tune_eval_summary_{train_size}.csv")
    subprocess.run(
    [
        "python", "hp_eval.py",
        "--input", os.path.join(out_hp_dir, f"hp_tune_cv{train_size}_summary.json"),
        "--out_csv", hp_eval_csv
    ],
    check=True
)


    # 2️⃣ Read the parser output and find the best config
    import pandas as pd, json
    best_config_path = os.path.join(out_hp_dir, "best_config.json")


    df = pd.read_csv(hp_eval_csv)
    if "mean_window_accuracy" not in df.columns or df.empty:
        log("⚠️ No valid HP evaluation results found.")
        continue


    best_row = df.loc[df["mean_window_accuracy"].idxmax()]
    best_config = {
        "lora_r": int(best_row["lora_r"]),
        "lora_alpha": int(best_row["lora_alpha"]),
        "lr": str(best_row["lr"])
  # or fetch from your HP grid if you add 'lr' to CSV
    }
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    log(f"✅ Best config selected: {best_config}")




    # ========================================================
    # STEP 3 — Train final model on full training data
    # ========================================================
    log("🧠 Training final model...")
    subprocess.run(
        [
            "python", TRAIN_SCRIPT,
            "--data", train_file,
            "--config", best_config_path,
            "--out", out_train_dir
        ],
        check=True
    )




    # ========================================================
    # STEP 4 — Evaluate on next 50 samples (test set)
    # ========================================================
    # ========================================================
    # STEP 4 — Evaluate on next 50 samples (test set)
    # ========================================================
    log("📊 Running evaluation parser (final results)...")

    test_results_path = os.path.join(out_train_dir, "test_metrics.json")
    pred_gold_path = os.path.join(out_train_dir, "pred_gold.jsonl")
    out_csv_path = os.path.join(out_train_dir, "hp_eval_test.csv")

    # Use the new evaluation script specialized for single JSONL results
    subprocess.run(
        [
            "python", "hp_eval_result.py",
            "--input", pred_gold_path,
            "--out_csv", out_csv_path
        ],
        check=True
    )







    # Extract summary stats
    if os.path.exists(os.path.join(out_train_dir, "hp_eval_test.csv")):
        df_test = pd.read_csv(os.path.join(out_train_dir, "hp_eval_test.csv"))
        test_acc = df_test["mean_window_accuracy"].mean() if not df_test.empty else 0
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": test_acc}, f, indent=2)
        log(f"✅ Test accuracy: {round(test_acc, 4)}")
    else:
        log("⚠️ No test results found.")






    # ========================================================
    # STEP 5 — Log metrics
    # ========================================================
    if os.path.exists(test_results_path):
        with open(test_results_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": None, "f1": None}

    log(f"✅ Completed TRAIN={train_size}: {metrics}")
    results.append({
        "train_size": train_size,
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "timestamp": timestamp
    })

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)
log(f"\n📁 Master run complete. Results saved to {RESULTS_CSV}")


print(f"Runtime: {time.time() - start:.4f} seconds")