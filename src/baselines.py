import numpy as np
import pandas as pd
import scipy.stats

from src.config import MAX_GOALS


class HistoricalFrequencyBaseline:
    """
    First baseline that predicts the match outcomes from training data, ignoring teams.
    
    This model should not work well, as it does not consider team strengths. 
    """

    def __init__(self):
        self.p_home_win = None
        self.p_draw = None
        self.p_away_win = None

    def fit(self, train_df):
        n = len(train_df)
        self.p_home_win = (train_df["FTR"] == "H").sum() / n # these should reflect the avgs in exploratory data analysis
        self.p_draw = (train_df["FTR"] == "D").sum() / n
        self.p_away_win = (train_df["FTR"] == "A").sum() / n
        return self

    def predict(self, test_df):
        out = test_df.copy()
        out["p_home_win"] = self.p_home_win
        out["p_draw"] = self.p_draw
        out["p_away_win"] = self.p_away_win
        return out