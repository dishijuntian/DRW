import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from config.config import Config


def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df, early_percentage: float):
    learner_ensembles = {}
    
    print("\nIndividual Slice Scores:")
    for learner_name in oof_preds:
        scores = {}
        for s in oof_preds[learner_name]:
            score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
            scores[s] = score
            print(f"  {learner_name} - {s}: {score:.4f}")
        
        total_score = sum(scores.values())
        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

        oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
        test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

        print(f"\n{learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")

        if score_weighted > score_simple:
            learner_ensembles[learner_name] = {
                "oof": oof_weighted,
                "test": test_weighted,
                "type": "weighted"
            }
        else:
            learner_ensembles[learner_name] = {
                "oof": oof_simple,
                "test": test_simple,
                "type": "simple"
            }

    final_oof = np.mean([le["oof"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]

    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")
    print(f"Ensemble types used: {[le['type'] for le in learner_ensembles.values()]}")

    filename = f"submission_early_{int(early_percentage*100)}pct.csv"
    submission_df["prediction"] = final_test
    submission_df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")