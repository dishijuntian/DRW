from config.config import Config
from feature_engineering.feature_engineering import feature_engineering
import pandas as pd

def load_data():
    RAW_FEATURES=[
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
        "bid_qty", "ask_qty"
    ]
    train_df=pd.read_parquet(Config.TRAIN_PATH,columns=RAW_FEATURES+[Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=RAW_FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df