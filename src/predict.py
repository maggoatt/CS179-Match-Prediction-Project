import numpy as np
import pandas as pd
import scipy.stats
import arviz as az
from src.config import MAX_GOALS, UNKNOWN_TEAM_ID


def _scoreline_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    """
    Build a (MAX_GOALS+1) x (MAX_GOALS+1) matrix M where
    M[x, y] = P(home_goals=x) * P(away_goals=y)
    under independent Poisson distributions.
    """
    goals = np.arange(MAX_GOALS + 1)
    p_home = scipy.stats.poisson.pmf(goals, lambda_home)
    p_away = scipy.stats.poisson.pmf(goals, lambda_away)
    return np.outer(p_home, p_away)


def _matrix_to_outcomes(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Given a (MAX_GOALS+1)^2 scoreline probability matrix, return
    (p_home_win, p_draw, p_away_win).

    matrix[x, y] = P(home=x, away=y)
    home wins when x > y → strictly lower triangle (np.tril k=-1)
    draw when x == y     → main diagonal
    away wins when x < y → strictly upper triangle (np.triu k=1)

    Normalizes to account for truncation at MAX_GOALS.
    """
    p_home_win = np.tril(matrix, k=-1).sum()
    p_draw = np.diag(matrix).sum()
    p_away_win = np.triu(matrix, k=1).sum()
    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total
