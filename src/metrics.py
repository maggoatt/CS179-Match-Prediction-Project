"""Probabilistic evaluation metrics for Home/Draw/Away match-outcome forecasts.

Lower is better.

Log loss, RPS, and Brier scores
"""

import numpy as np
import pandas as pd

# Ordinal outcome scale (home win -> draw -> away win) and the matching
# probability columns produced by every model's `predict`.
OUTCOMES = ["H", "D", "A"]
PROB_COLUMNS = ["ProbHomeWin", "ProbDraw", "ProbAwayWin"]


def _probs_and_onehot(preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Pull the predicted probability matrix and one-hot true outcomes out of a
    predictions DataFrame, with columns aligned to ``OUTCOMES``."""
    probs = preds[list(prob_cols)].to_numpy(dtype=float)
    actual = preds[result_col].to_numpy()
    onehot = np.stack([(actual == o) for o in OUTCOMES], axis=1).astype(float)
    return probs, actual, onehot


def accuracy(preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Fraction of matches whose most-likely predicted outcome was correct."""
    probs, actual, _ = _probs_and_onehot(preds, result_col, prob_cols)
    predicted = np.array(OUTCOMES)[probs.argmax(axis=1)]
    return float((predicted == actual).mean())


def log_loss(preds, result_col="Result", prob_cols=PROB_COLUMNS, eps=1e-15):
    """Mean negative log-likelihood of the realised outcomes (cross-entropy)."""
    probs, _, onehot = _probs_and_onehot(preds, result_col, prob_cols)
    prob_true = (probs * onehot).sum(axis=1)
    return float(-np.log(np.clip(prob_true, eps, 1.0)).mean())


def ranked_probability_score(preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Mean RPS. Order-aware: backing a draw when the result was an away win is
    penalised less than backing a home win, because H-D-A is an ordinal scale."""
    probs, _, onehot = _probs_and_onehot(preds, result_col, prob_cols)
    cum_pred = np.cumsum(probs, axis=1)
    cum_obs = np.cumsum(onehot, axis=1)
    r = probs.shape[1]
    return float((((cum_pred - cum_obs) ** 2).sum(axis=1) / (r - 1)).mean())


def brier_score(preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Multiclass Brier score: mean squared error of the probability vector."""
    probs, _, onehot = _probs_and_onehot(preds, result_col, prob_cols)
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def evaluate(preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Return accuracy plus the three probabilistic scores as a dict.

    Accuracy is included so a single call yields a complete, comparable row;
    the probabilistic scores are the metrics that go beyond it.
    """
    return {
        "Accuracy": accuracy(preds, result_col, prob_cols),
        "Log Loss": log_loss(preds, result_col, prob_cols),
        "RPS": ranked_probability_score(preds, result_col, prob_cols),
        "Brier": brier_score(preds, result_col, prob_cols),
    }


def evaluate_many(named_preds, result_col="Result", prob_cols=PROB_COLUMNS):
    """Score several models at once.

    ``named_preds`` maps a model name to its predictions DataFrame; returns a
    DataFrame with one row per model and one column per metric.
    """
    return pd.DataFrame(
        {name: evaluate(preds, result_col, prob_cols) for name, preds in named_preds.items()}
    ).T
