import sys
from config.config import Config
from load_data.load_data import load_data
from train_and_evaluate.train_and_evaluate import train_and_evaluate
from ensemble_and_submit.ensemble_and_submi import ensemble_and_submit

# ========== CONFIGURATION - CHANGE THIS VALUE ==========
EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45
# ======================================================

if __name__ == "__main__":
    print(f"\nRunning with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df, EARLY_PERCENTAGE)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df, EARLY_PERCENTAGE)