"""
A module for a Bayesian team-strength model (and various ablations) for Premier League match outcome predictions.
"""
from dataclasses import dataclass
from typing import cast

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from torch import Tensor
from torch import device as Device
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

from src.config import MAX_GOALS


@dataclass(frozen = True, kw_only = True, slots = True)
class SVIConfig:
    """
    Represents a configuration for SVI.
    """
    record_loss_every_n_steps: int = 100
    num_steps: int = 10_000
    learning_rate: float = 0.0025
    num_particles: int = 5
    clip_norm: float = 5.0

@dataclass(frozen = True, kw_only = True, slots = True)
class BayesianPoissonModelConfig:
    """
    Represents a configuration for the Bayesian Poisson team-strength model.
    """
    seed: int
    device: Device
    svi: SVIConfig
    ablate_attack: bool
    ablate_defense: bool
    ablate_home_team_advantage: bool
    num_posterior_samples_per_match: int
    maximum_goals_in_match: int = MAX_GOALS


class BayesianPoissonModel:
    """
    A Bayesian Poisson team-strength model predicting match outcomes based on Poisson distributions over home-team goals and away-team goals.
    
    Given a home team A and an away team B:
    - The Poisson distribution over home-team goals is parameterized by λ_home
        - λ_home = exp(μ + home_team_advantage + home_attacking_strength_A - away_defensive_strength_B)
            - μ represents a general scoring bias for any team
            - home_team_advantage represents a general home-team advantage bias for any team
            - home_attacking_strength_i represents the mean-centered latent attacking strength of team i
            - away_defensive_strength_j represents the mean-centered latent defensive strength of team j
    - The Poisson distribution over away-team goals is parameterized by λ_away
        - λ_away = exp(μ + home_attacking_strength_B - away_defensive_strength_A)
            - μ represents a general scoring bias for any team
            - home_attacking_strengh_i represents the mean-centered latent attacking strength of team i
            - away_defensive_strength_j represents the mean-centered latent defensive strength of team j
    - Each parameter (e.g. μ, h, attacking_strength_i, defensive_strength_j) is represented by some prior distribution
    - Given training data, the model employs SVI to infer posterior distributions for the parameters

    Because the model does not represent each parameter with a point estimate and instead represents each parameter with a distribution, the model is Bayesian.
    """
    def __init__(self, *, config: BayesianPoissonModelConfig) -> None:
        """
        Initializes the Bayesian Poisson team-strength model.
        
        :param config: A configuration for the Bayesian Poisson team-strength model.
        :return: None.
        """
        self.config = config
        self.guide: AutoNormal | None = None
        self.num_teams: int | None = None

    def fit(self, train_df: pd.DataFrame) -> "BayesianPoissonModel":
        """
        Fits the Bayesian Poisson team-strength model to the training data via SVI.
        
        :param train_df: A DataFrame object storing historical Premier League match data to train from.
        :return: The fitted model instance.
        """
        pyro.clear_param_store()
        pyro.set_rng_seed(self.config.seed)
        
        self.num_teams = len(set(train_df["HomeTeam"].unique()) | set(train_df["AwayTeam"].unique()))
        home_team_ids = self._series_to_tensor(train_df["HomeTeamID"])
        away_team_ids = self._series_to_tensor(train_df["AwayTeamID"])
        observed_home_team_goals = self._series_to_tensor(train_df["HomeGoals"])
        observed_away_team_goals = self._series_to_tensor(train_df["AwayGoals"])
        
        self.guide = AutoNormal(self._model)
        optimizer = ClippedAdam({"lr": self.config.svi.learning_rate,
                                 "clip_norm": self.config.svi.clip_norm})
        svi = SVI(self._model, self.guide, optimizer, loss = Trace_ELBO(num_particles = self.config.svi.num_particles))
        for step in range(1, self.config.svi.num_steps + 1):
            loss = svi.step(home_team_ids, away_team_ids, observed_home_team_goals, observed_away_team_goals)
            loss = cast(float, loss)
            self._record_loss(step, loss)
        return self

    def _series_to_tensor(self, series: pd.Series) -> Tensor:
        return torch.tensor(series.to_numpy(),
                            dtype = torch.long, device = self.config.device)

    def _model(self,
               home_team_ids: Tensor,
               away_team_ids: Tensor,
               observed_home_goals: Tensor,
               observed_away_goals: Tensor) -> None:
        mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
        home_team_advantage = self._sample_home_team_advantage()
        home_attacking_strengths = self._sample_team_strengths("home_attacking_strengths", self.config.ablate_attack)
        away_attacking_strengths = self._sample_team_strengths("away_attacking_strengths", self.config.ablate_attack)
        home_defensive_strengths = self._sample_team_strengths("home_defensive_strengths", self.config.ablate_defense)
        away_defensive_strengths = self._sample_team_strengths("away_defensive_strengths", self.config.ablate_defense)

        lambda_home = torch.exp(mu + home_team_advantage +
                                home_attacking_strengths[home_team_ids] - away_defensive_strengths[away_team_ids])
        lambda_away = torch.exp(mu + away_attacking_strengths[away_team_ids] - home_defensive_strengths[home_team_ids])

        with pyro.plate("matches", len(home_team_ids)):
            pyro.sample("home_goals", dist.Poisson(lambda_home),
                        obs = observed_home_goals)
            pyro.sample("away_goals", dist.Poisson(lambda_away),
                        obs = observed_away_goals)

    def _sample_home_team_advantage(self) -> Tensor:
        if self.config.ablate_home_team_advantage:
            return torch.zeros((), device = self.config.device)
        return pyro.sample("home_team_advantage", dist.Normal(0.0, 0.5))

    def _sample_team_strengths(self,
                               latent_variable_name: str,
                               ablate: bool) -> Tensor:
        if self.num_teams is None:
            raise ValueError("Cannot generate latent team strength samples. "
                             "The model does not know how many teams exist in the training dataset. "
                             "Call `fit` before calling `_sample_team_strengths`.")
        if ablate:
            return torch.zeros(self.num_teams, device = self.config.device)
        raw_team_strengths = pyro.sample(latent_variable_name,
                                         dist.Normal(0.0, 1.0).expand([self.num_teams]).to_event(1))
        return raw_team_strengths - raw_team_strengths.mean()

    def _record_loss(self, step: int, loss: float) -> None:
        if step % self.config.svi.record_loss_every_n_steps == 0:
            print(f"Step {step}: Loss = {loss:.3f}")
        
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the probabilities of home wins, draws, and away wins for the test data using the fitted Bayesian Poisson team-strength model.
        
        :param test_df: A DataFrame object storing historical Premier League match data to predict on.
        :return: A DataFrame object containing the original test data along with the predicted probabilities for each match outcome.
        """
        if self.guide is None or self.num_teams is None:
            raise ValueError("Cannot predict before fitting the model. Call `fit` before calling `predict`.")
        if self.config.num_posterior_samples_per_match < 1:
            raise ValueError("The number of posterior samples per test match must be positive.")
        if self.config.maximum_goals_in_match < 1:
            raise ValueError("The maximum goals in a match (for truncating the Poisson distribution over goals) must be positive.")

        predict_df = test_df.copy()
        home_team_ids = self._series_to_tensor(test_df["HomeTeamID"])
        away_team_ids = self._series_to_tensor(test_df["AwayTeamID"])
        dummy_home_goals = torch.zeros_like(home_team_ids)
        dummy_away_goals = torch.zeros_like(away_team_ids)
    
        probs = []
        with torch.no_grad():
            for _, row in predict_df.iterrows():
                probs_per_test_match = []
                for _ in range(self.config.num_posterior_samples_per_match):
                    home_team_id, away_team_id = int(row["HomeTeamID"]), int(row["AwayTeamID"])
                    posterior_sample = self.guide(home_team_ids,
                                                  away_team_ids,
                                                  dummy_home_goals,
                                                  dummy_away_goals)
                    mu = posterior_sample["mu"]
                    home_team_advantage = self._get_posterior_home_team_advantage(posterior_sample)
                    home_attacking_strengths = self._get_posterior_team_strengths(posterior_sample, "home_attacking_strengths", self.config.ablate_attack)
                    away_attacking_strengths = self._get_posterior_team_strengths(posterior_sample, "away_attacking_strengths", self.config.ablate_attack)
                    home_defensive_strengths = self._get_posterior_team_strengths(posterior_sample, "home_defensive_strengths", self.config.ablate_defense)
                    away_defensive_strengths = self._get_posterior_team_strengths(posterior_sample, "away_defensive_strengths", self.config.ablate_defense)

                    lambda_home = torch.exp(mu + home_team_advantage + self._lookup_team_strength(home_attacking_strengths, home_team_id) - self._lookup_team_strength(away_defensive_strengths, away_team_id))
                    lambda_away = torch.exp(mu + self._lookup_team_strength(away_attacking_strengths, away_team_id) - self._lookup_team_strength(home_defensive_strengths, home_team_id))
        
                    sampled_probs = self._compute_outcome_probabilities(lambda_home, lambda_away)
                    probs_per_test_match.append(sampled_probs)
                    
                probs.append(torch.stack(probs_per_test_match).mean(dim = 0))
            probs = torch.stack(probs).detach().cpu().numpy()
            predict_df[["ProbHomeWin", "ProbDraw", "ProbAwayWin"]] = probs
        return predict_df

    def _lookup_team_strength(self, strengths: Tensor, team_id: int) -> Tensor:
        if team_id < 0:
            return torch.zeros((), device = self.config.device, dtype = strengths.dtype)
        return strengths[team_id]

    def _get_posterior_home_team_advantage(self, posterior_sample: dict[str, Tensor]) -> Tensor:
        if self.config.ablate_home_team_advantage:
            return torch.zeros((), device = self.config.device)
        return posterior_sample["home_team_advantage"]

    def _get_posterior_team_strengths(self,
                                      posterior_sample: dict[str, Tensor],
                                      latent_variable_name: str,
                                      ablate: bool) -> Tensor:
        if self.num_teams is None:
            raise ValueError("Cannot generate latent team strength samples. "
                             "The model does not know how many teams exist in the training dataset. "
                             "Call `fit` before calling `_get_posterior_team_strengths`.")
        if ablate:
            return torch.zeros(self.num_teams, device = self.config.device)
        raw_team_strengths = posterior_sample[latent_variable_name]
        return raw_team_strengths - raw_team_strengths.mean()

    def _compute_outcome_probabilities(self, lambda_home: Tensor, lambda_away: Tensor) -> Tensor:
        prob_home_goals = self._compute_goal_probs(lambda_home)
        prob_away_goals = self._compute_goal_probs(lambda_away)
        prob_scoreline = prob_home_goals[:, None] * prob_away_goals[None, :]

        prob_home_win = torch.tril(prob_scoreline, diagonal = -1).sum()
        prob_draw = torch.diag(prob_scoreline).sum()
        prob_away_win = torch.triu(prob_scoreline, diagonal = 1).sum()

        return torch.stack([prob_home_win, prob_draw, prob_away_win])
        

    def _compute_goal_probs(self, goal_rate: Tensor) -> Tensor:
        possible_goals = torch.arange(self.config.maximum_goals_in_match + 1,
                                      dtype = goal_rate.dtype,
                                      device=self.config.device)
        prob_goals = dist.Poisson(goal_rate).log_prob(possible_goals).exp()
        return prob_goals / prob_goals.sum()
