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


def predict_all(
    test_df: pd.DataFrame,
    idata: az.InferenceData,
) -> pd.DataFrame:
    """
    For each match in test_df, average scoreline matrices over all Markov Chain Monte Carlo
    samples to get outcome probabilities.

    For each match (home=i, away=j) and each posterior sample s:
      lambda_home_s = exp(mu_s + home_adv_s + attack_s[i] - defense_s[j])
      lambda_away_s = exp(mu_s + attack_s[j] - defense_s[i])

    We build a scoreline matrix per sample, average across all samples,
    then read off p_home_win / p_draw / p_away_win from the averaged matrix.

    Unknown teams (id == -1) fall back to league-average lambdas:
      lambda_home = exp(mu + home_adv),  lambda_away = exp(mu)
    """
    post = idata.posterior
    n_chains, n_draws = post["attack"].shape[:2]
    n_samples = n_chains * n_draws

    attack_samples   = post["attack"].values.reshape(n_samples, -1)   # (S, n_teams)
    defense_samples  = post["defense"].values.reshape(n_samples, -1)  # (S, n_teams)
    mu_samples       = post["mu"].values.reshape(n_samples)            # (S,)
    home_adv_samples = post["home_adv"].values.reshape(n_samples)      # (S,)

    goals = np.arange(MAX_GOALS + 1)

    results = []
    for _, row in test_df.iterrows():
        hi = int(row["home_team_id"])
        ai = int(row["away_team_id"])

        if hi == UNKNOWN_TEAM_ID or ai == UNKNOWN_TEAM_ID:
            lh_samples = np.exp(mu_samples + home_adv_samples)
            la_samples = np.exp(mu_samples)
        else:
            lh_samples = np.exp(mu_samples + home_adv_samples + attack_samples[:, hi] - defense_samples[:, ai])
            la_samples = np.exp(mu_samples + attack_samples[:, ai] - defense_samples[:, hi])

        # Shape: (S, MAX_GOALS+1) — Poisson pmf for each sample
        p_home_per_sample = scipy.stats.poisson.pmf(goals[np.newaxis, :], lh_samples[:, np.newaxis])
        p_away_per_sample = scipy.stats.poisson.pmf(goals[np.newaxis, :], la_samples[:, np.newaxis])

        # Shape: (S, MAX_GOALS+1, MAX_GOALS+1)
        matrices = p_home_per_sample[:, :, np.newaxis] * p_away_per_sample[:, np.newaxis, :]

        avg_matrix = matrices.mean(axis=0)  # (MAX_GOALS+1, MAX_GOALS+1)

        p_hw, p_d, p_aw = _matrix_to_outcomes(avg_matrix)
        mode_idx = np.unravel_index(avg_matrix.argmax(), avg_matrix.shape)

        results.append({
            **row.to_dict(),
            "p_home_win":     p_hw,
            "p_draw":         p_d,
            "p_away_win":     p_aw,
            "exp_home_goals": float(lh_samples.mean()),
            "exp_away_goals": float(la_samples.mean()),
            "mode_scoreline": f"{mode_idx[0]}-{mode_idx[1]}",
        })

    return pd.DataFrame(results)


def predict_match(
    home_team: str,
    away_team: str,
    idata: az.InferenceData,
    team_mapping: dict,
) -> dict:
    """
    Convenience function: predict a single match by team name.
    Returns p_home_win, p_draw, p_away_win, exp_home_goals,
    exp_away_goals, and mode_scoreline.
    """
    hi = team_mapping.get(home_team, UNKNOWN_TEAM_ID)
    ai = team_mapping.get(away_team, UNKNOWN_TEAM_ID)

    stub = pd.DataFrame([{
        "Date": None, "HomeTeam": home_team, "AwayTeam": away_team,
        "FTHG": 0, "FTAG": 0, "FTR": "H", "season": "unknown",
        "home_team_id": hi, "away_team_id": ai,
    }])

    row = predict_all(stub, idata).iloc[0]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win": row["p_home_win"],
        "p_draw": row["p_draw"],
        "p_away_win": row["p_away_win"],
        "exp_home_goals": row["exp_home_goals"],
        "exp_away_goals": row["exp_away_goals"],
        "mode_scoreline": row["mode_scoreline"],
    }
