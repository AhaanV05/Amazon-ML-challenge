"""Candidate-set size vs recall ceiling for filter settings. Run: python -m src.filter_sweep TAG"""
import itertools
import sys

import numpy as np
import pandas as pd

from .config import WORK_DIR
from .metric import oracle_ceiling
from .train import fold_of, truth_sets


def main(tag: str = "work20") -> None:
    s = pd.read_parquet(WORK_DIR / "stage" / f"{tag}_filter_scores.parquet")
    entities = pd.read_parquet(WORK_DIR / "stage" / f"{tag}_entities.parquet")["q_id"].to_numpy()
    val_ids = entities[fold_of(entities) == 4].tolist()
    truth = truth_sets(set(val_ids))
    s = s[s["fold"] == 4].reset_index(drop=True)
    s["rank_d"] = s.groupby(["source", "d_id"])["p_filter"].rank(ascending=False, method="first")
    s["rank_q"] = s.groupby(["source", "q_id"])["p_filter"].rank(ascending=False, method="first")
    n_true = sum(len(v) for v in truth.values())
    rows = []
    for tau, r, k in itertools.product((0.01, 0.02, 0.05, 0.1, 0.2), (1, 2, 3), (3, 4, 6, 8)):
        kept = s[(s["p_filter"] >= tau) & (s["rank_d"] <= r) & (s["rank_q"] <= k)]
        per_q = kept.groupby("q_id").size().reindex(val_ids, fill_value=0)
        found = kept[kept["label"]].groupby("q_id")["d_id"].agg(set).to_dict()
        rows.append({"tau": tau, "r": r, "k_max": k, "mean_cands": per_q.mean(), "p95": per_q.quantile(0.95),
                     "pair_recall": kept["label"].sum() / n_true, "oracle": oracle_ceiling(found, truth, val_ids)})
    out = pd.DataFrame(rows).sort_values("mean_cands")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv(WORK_DIR / "stage" / f"{tag}_filter_sweep.csv", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "work20")
