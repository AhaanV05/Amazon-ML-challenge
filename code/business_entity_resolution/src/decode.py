"""Turn pair probabilities into per-entity match lists."""
import numpy as np
import pandas as pd

MIN_P = 0.02


def one_owner(pairs: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    """Each S2/S3 record belongs to at most one S1 entity (0 violations in train): keep its best pair."""
    best = pairs.groupby("d_id")[p_col].transform("max")
    return pairs[pairs[p_col] >= best]


def _poisson_binomial(ps: np.ndarray) -> np.ndarray:
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for i, p in enumerate(ps):
        dist[1:i + 2] = dist[1:i + 2] * (1 - p) + dist[:i + 1] * p
        dist[0] *= 1 - p
    return dist


def best_k(ps: np.ndarray, extra_missed: float = 0.0) -> int:
    """ps sorted descending. Size of the top-k prefix maximizing expected F0.5 (k=0 is the empty list)."""
    best, best_score = 0, -1.0
    for k in range(len(ps) + 1):
        top, rest = _poisson_binomial(ps[:k]), _poisson_binomial(ps[k:])
        a = np.arange(len(top))[:, None]
        b = np.arange(len(rest))[None, :] + extra_missed
        n_true = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(k == 0, (n_true == 0).astype(float), np.where(a > 0, 1.25 * a / (0.25 * n_true + k), 0.0))
        score = float((top[:, None] * rest[None, :] * f).sum())
        if score > best_score:
            best, best_score = k, score
    return best


def decode_expected_f(pairs: pd.DataFrame, p_col: str = "p") -> dict[str, list[str]]:
    kept = pairs[pairs[p_col] >= MIN_P].sort_values(["q_id", p_col], ascending=[True, False])
    out = {}
    for q_id, g in kept.groupby("q_id", sort=False):
        ps = g[p_col].to_numpy()
        k = best_k(ps)
        if k:
            out[q_id] = g["d_id"].to_numpy()[:k].tolist()
    return out


def decode_threshold(pairs: pd.DataFrame, threshold: float, p_col: str = "p") -> dict[str, list[str]]:
    kept = pairs[pairs[p_col] >= threshold]
    return kept.groupby("q_id")["d_id"].agg(list).to_dict()
