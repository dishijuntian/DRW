import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr



def create_time_decay_weights(n:int,decay:float=0.9,reverse:bool=False)->np.ndarray:
    positions=np.arange(n)
    # if reverse:
    #     normalized = 1.0 - (positions / (n - 1))
    # else:
    #     normalized = positions / (n - 1)
    normalized = positions/(n-1)
    weights=decay**(1.0-normalized)
    return weights*n/weights.sum()
    