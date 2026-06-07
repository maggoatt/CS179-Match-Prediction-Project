import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path


def build_and_fit(
    train_df: pd.DataFrame,
    team_mapping: dict,
    draws: int = 1000,
    tune: int = 500,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Fit the full Bayesian attack-defense Poisson model on training data.

    Model:
        mu, home_adv         ~ Normal(0, 1)
        sigma_att, sigma_def ~ HalfNormal(1)
        attack_raw[i]  ~ Normal(0, sigma_att)   shape=(n_teams,)
        defense_raw[i] ~ Normal(0, sigma_def)   shape=(n_teams,)
        attack  = attack_raw  - mean(attack_raw)   # centering for identifiability
        defense = defense_raw - mean(defense_raw)

        log_lambda_home = mu + home_adv + attack[home_id] - defense[away_id]
        log_lambda_away = mu            + attack[away_id]  - defense[home_id]

        FTHG ~ Poisson(exp(log_lambda_home))
        FTAG ~ Poisson(exp(log_lambda_away))

    Saves trace to POSTERIOR_DIR / "trace.nc" and returns the InferenceData.
    """
    from src.config import POSTERIOR_DIR

    n_teams    = len(team_mapping)
    home_ids   = train_df["home_team_id"].values
    away_ids   = train_df["away_team_id"].values
    home_goals = train_df["FTHG"].values
    away_goals = train_df["FTAG"].values

    valid = (home_ids >= 0) & (away_ids >= 0)
    home_ids   = home_ids[valid]
    away_ids   = away_ids[valid]
    home_goals = home_goals[valid]
    away_goals = away_goals[valid]

    with pm.Model() as model:
        mu       = pm.Normal("mu",       mu=0, sigma=1)
        home_adv = pm.Normal("home_adv", mu=0, sigma=1)

        sigma_att = pm.HalfNormal("sigma_att", sigma=1)
        sigma_def = pm.HalfNormal("sigma_def", sigma=1)

        attack_raw  = pm.Normal("attack_raw",  mu=0, sigma=sigma_att, shape=n_teams)
        defense_raw = pm.Normal("defense_raw", mu=0, sigma=sigma_def, shape=n_teams)

        attack  = pm.Deterministic("attack",  attack_raw  - pt.mean(attack_raw))
        defense = pm.Deterministic("defense", defense_raw - pt.mean(defense_raw))

        log_lh = mu + home_adv + attack[home_ids] - defense[away_ids]
        log_la = mu + attack[away_ids] - defense[home_ids]

        pm.Poisson("home_goals", mu=pm.math.exp(log_lh), observed=home_goals)
        pm.Poisson("away_goals", mu=pm.math.exp(log_la), observed=away_goals)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, POSTERIOR_DIR / "trace.nc")
    return idata


def load_trace(filename: str = "trace.nc") -> az.InferenceData:
    """Load a previously saved MCMC trace from POSTERIOR_DIR."""
    from src.config import POSTERIOR_DIR
    return az.from_netcdf(POSTERIOR_DIR / filename)
