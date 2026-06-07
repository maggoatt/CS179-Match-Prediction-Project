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
    
class TeamPoissonBaseline:
    """
    Per-team average goals -> Poisson-based expected goals -> score matrix -> outcome probs

    For match (home=i, away=j):
        lambda_home = att_home[i] * def_away[j] / league_avg_home_goals
        lambda_away = att_away[j] * def_home[i] / league_avg_away_goals
    """

    def __init__(self):
        self.avg_home_goals = None
        self.avg_away_goals = None
        self.home_stats = None
        self.away_stats = None
        self.default_home_attack = None
        self.default_away_attack = None
        self.default_home_defense = None
        self.default_away_defense = None

    def fit(self, train_df):
        self.avg_home_goals = train_df["FTHG"].mean()
        self.avg_away_goals = train_df["FTAG"].mean()

        self.home_stats = train_df.groupby("HomeTeam").agg(home_attack=("FTHG", "mean"), home_defense=("FTAG", "mean"))
        self.away_stats = train_df.groupby("AwayTeam").agg(away_attack=("FTAG", "mean"), away_defense=("FTHG", "mean"))

        # for teams with no training data
        self.default_home_attack = self.avg_home_goals
        self.default_away_attack = self.avg_away_goals
        self.default_home_defense = self.avg_away_goals
        self.default_away_defense = self.avg_home_goals
        return self

    def _outcome_probs(self, lambda_home, lambda_away):
        goals = np.arange(MAX_GOALS + 1)
        p_home = scipy.stats.poisson.pmf(goals, lambda_home)
        p_away = scipy.stats.poisson.pmf(goals, lambda_away)
        matrix = np.outer(p_home, p_away)

        p_home_win = np.tril(matrix, k=-1).sum()
        p_draw = np.diag(matrix).sum()
        p_away_win = np.triu(matrix, k=1).sum()

        total = p_home_win + p_draw + p_away_win
        return p_home_win / total, p_draw / total, p_away_win / total

    def predict(self, test_df):
        out = test_df.copy()
        probs = []
        for _, row in out.iterrows():
            i, j = row["HomeTeam"], row["AwayTeam"]

            att_i_home = self.home_stats["home_attack"].get(i, self.default_home_attack)
            def_j_away = self.away_stats["away_defense"].get(j, self.default_away_defense)
            att_j_away = self.away_stats["away_attack"].get(j, self.default_away_attack)
            def_i_home = self.home_stats["home_defense"].get(i, self.default_home_defense)

            lambda_home = att_i_home * def_j_away / self.avg_home_goals
            lambda_away = att_j_away * def_i_home / self.avg_away_goals

            probs.append(self._outcome_probs(lambda_home, lambda_away))

        out[["p_home_win", "p_draw", "p_away_win"]] = probs
        return out