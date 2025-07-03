from xgboost import XGBRegressor
class Config:
    TRAIN_PATH = "E:/GIT PROJECT/drw/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "E:/GIT PROJECT/drw//kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "E:/GIT PROJECT/drw//kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
        "bid_qty", "ask_qty",
        "bid_ask_interaction", "bid_buy_interaction", "bid_sell_interaction",
        "ask_buy_interaction", "ask_sell_interaction", "buy_sell_interaction",
        "spread_indicator", "total_liquidity", "liquidity_imbalance",
        "relative_spread", "depth_ratio", "volume_weighted_buy",
        "volume_weighted_sell", "volume_weighted_bid", "volume_weighted_ask",
        "bid_ask_ratio", "trade_direction_ratio", "order_flow_imbalance",
        "buying_pressure", "net_buy_volume", "trade_intensity",
        "avg_trade_size", "net_trade_flow", "volume_participation",
        "market_activity", "realized_volatility_proxy",
        "normalized_buy_volume", "normalized_sell_volume",
        "liquidity_adjusted_imbalance", "pressure_spread_interaction",
        "bid_skew", "ask_skew"
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

XGB_PARAMS = {
    'tree_method': 'hist', 
    'device': 'gpu',
    'n_jobs': -1,
    'colsample_bytree': 0.4111224922845363, 
    'colsample_bynode': 0.28869302181383194,
    'gamma': 1.4665430311056709, 
    'learning_rate': 0.014053505540364681, 
    'max_depth': 7, 
    'max_leaves': 40, 
    'n_estimators': 500,
    'reg_alpha': 27.791606770656145, 
    'reg_lambda': 84.90603428439086,
    'subsample': 0.06567,
    'verbosity': 0,
    'random_state': Config.RANDOM_STATE
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]
