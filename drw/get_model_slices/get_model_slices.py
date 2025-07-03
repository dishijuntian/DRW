import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr

def get_model_slices(n_samples: int, early_percentage: float):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(early_percentage*100)}pct", "type": "early", "cutoff": int(early_percentage * n_samples)},
    ]
