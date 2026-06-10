"""
A module for baseline models for Premier League match outcome prediction.
"""
import numpy as np
import pandas as pd
import scipy.stats

from src.config import MAX_GOALS


class HistoricalFrequencyBaseline:
    """
    A baseline model predicting match outcomes based on the historical frequencies of home wins, draws, and away wins in the training data.
    
    The model ignores team-specific information (and thus should not perform well).
    """
    def __init__(self) -> None:
        """
        Initializes the historical frequency baseline model.
        
        :return: None.
        """
        self.prob_home_win: float | None = None
        self.prob_draw: float | None = None
        self.prob_away_win: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "HistoricalFrequencyBaseline":
        """
        Fits the historical frequency baseline model to the training data by calculating the probabilities of home wins, draws, and away wins.
        
        :param train_df: A DataFrame object storing historical Premier League match data to train from.
        :return: The fitted model instance.
        """
        n = len(train_df)
        # NOTE: These should reflect the averages in baseline evaluation
        self.prob_home_win = (train_df["Result"] == "H").sum() / n
        self.prob_draw = (train_df["Result"] == "D").sum() / n
        self.prob_away_win = (train_df["Result"] == "A").sum() / n
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the probabilities of home wins, draws, and away wins for the test data using the historical frequencies calculated during fitting.
        
        :param test_df: A DataFrame object storing historical Premier League match data to predict on.
        :return: A DataFrame object containing the original test data along with the predicted probabilities for each match outcome.
        """
        predict_df = test_df.copy()
        predict_df["ProbHomeWin"] = self.prob_home_win
        predict_df["ProbDraw"] = self.prob_draw
        predict_df["ProbAwayWin"] = self.prob_away_win
        return predict_df


class NonBayesianPoissonBaseline:
    """
    A non-Bayesian baseline model predicting match outcomes based on Poisson distributions over home goals and away goals.
    
    Given a home team A and an away team B:

    - The Poisson distribution over home goals is parameterized by λ_home, where λ_home = HomeAttack_A * AwayDefense_B / Average(HomeAttack). HomeAttack_i is estimated as the average # of home goals scored by team i, and AwayDefense_j is estimated as the average # of away goals conceded by team j.
    - The Poisson distribution over away goals is parameterized by λ_away, where λ_away = AwayAttack_B * HomeDefense_A / Average(AwayAttack). AwayAttack_i is estimated as the average # of away goals scored by team i, and HomeDefense_j is estimated as the average # of home goals conceded by team j.
    
    This baseline model is non-Bayesian: the model does not maintain a team's latent attacking and defending strengths with priors and posteriors.
    """

    def __init__(self) -> None:
        """
        Initializes the non-Bayesian Poisson baseline model.
        
        :return: None.
        """
        self.home_strength: pd.DataFrame | None = None
        self.away_strength: pd.DataFrame | None = None
        self.avg_home_goals: float | None = None
        self.avg_away_goals: float | None = None
        self.default_home_attack: float | None = None
        self.default_home_defense: float | None = None
        self.default_away_attack: float | None = None
        self.default_away_defense: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "NonBayesianPoissonBaseline":
        """
        Fits the non-Bayesian Poisson baseline model to the training data by calculating the average number of home goals scored and conceded by a team and 
        the average number of away goals scored and conceded by a team.
        
        :param train_df: A DataFrame object storing historical Premier League match data to train from.
        :return: The fitted model instance.
        """
        self.home_strength = train_df.groupby("HomeTeam").agg(HomeAttack = ("HomeGoals", "mean"), HomeDefense = ("AwayGoals", "mean"))[["HomeAttack", "HomeDefense"]]
        self.away_strength = train_df.groupby("AwayTeam").agg(AwayAttack = ("AwayGoals", "mean"), AwayDefense = ("HomeGoals", "mean"))[["AwayAttack", "AwayDefense"]]

        # For teams with no training data
        self.avg_home_goals = train_df["HomeGoals"].mean()
        self.avg_away_goals = train_df["AwayGoals"].mean()
        self.default_home_attack = self.avg_home_goals
        self.default_home_defense = self.avg_away_goals
        self.default_away_attack = self.avg_away_goals
        self.default_away_defense = self.avg_home_goals
        return self

    def _outcome_probs(self, lambda_home: float, lambda_away: float) -> tuple[float, float, float]:
        """
        Predicts the probabilities of the home team winning, drawing, and the away team winning 
        from the parameter of the Poisson distribution over home team goals and 
        the parameter of the Poisson distribution over away team goals.
        
        :param lambda_home: The parameter of the Poisson distribution over home team goals.
        :param lambda_away: The parameter of the Poisson distribution over away team goals.
        :return: The probabilities of the home team winning, drawing, and the away team winning.
        """
        goals = np.arange(MAX_GOALS + 1)
        prob_home_goals = scipy.stats.poisson.pmf(goals, lambda_home)
        prob_away_goals = scipy.stats.poisson.pmf(goals, lambda_away)
        prob_scoreline = np.outer(prob_home_goals, prob_away_goals)

        prob_home_win = np.tril(prob_scoreline, k = -1).sum()
        prob_draw = np.diag(prob_scoreline).sum()
        prob_away_win = np.triu(prob_scoreline, k = 1).sum()

        total = prob_home_win + prob_draw + prob_away_win
        return prob_home_win / total, prob_draw / total, prob_away_win / total

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the probabilities of home wins, draws, and away wins for the test data using the home and away strength estimates calculated during fitting.
        
        :param test_df: A DataFrame object storing historical Premier League match data to predict on.
        :return: A DataFrame object containing the original test data along with the predicted probabilities for each match outcome.
        """
        if self.home_strength is None or self.away_strength is None:
            raise ValueError("The model has no attacking or defending team-strength estimates. Call `fit` before calling `predict`.")
        predict_df = test_df.copy()
        probs = []
        for _, row in predict_df.iterrows():
            a, b = row["HomeTeam"], row["AwayTeam"]
            
            att_a_home = self.home_strength["HomeAttack"].get(a, self.default_home_attack)
            def_b_away = self.away_strength["AwayDefense"].get(b, self.default_away_defense)
            att_b_away = self.away_strength["AwayAttack"].get(b, self.default_away_attack)
            def_a_home = self.home_strength["HomeDefense"].get(a, self.default_home_defense)

            lambda_home = att_a_home * def_b_away / self.avg_home_goals
            lambda_away = att_b_away * def_a_home / self.avg_away_goals

            probs.append(self._outcome_probs(lambda_home, lambda_away))

        predict_df[["ProbHomeWin", "ProbDraw", "ProbAwayWin"]] = probs
        return predict_df
