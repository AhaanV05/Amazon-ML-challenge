"""Measure retrieval recall on train.

Run: python -m src.eval_retrieval COUNTRY MAX_QUERIES MAX_DF K_REV
With MAX_QUERIES < all, forward recall is exact but reverse recall is optimistic (fewer competing queries).
"""
import sys
import time

import numpy as np
import pandas as pd

from .config import SEED
from .data import ground_truth_pairs
from .prepare import load_norm
from .retrieve import ROUTES, retrieve

KS = (1, 3, 5, 10, 20, 50)


def owners(docs: pd.DataFrame, s1: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """For each doc row, the row index of its true S1 entity among s1 (or -1)."""
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    owner = pairs.set_index("match_id")["s1_id"].map(s1_pos)
    return docs["entity_id"].map(owner).fillna(-1).astype(np.int64).to_numpy()


def main(country: str, max_queries: int, max_df: float, k_rev: int) -> None:
    pairs = ground_truth_pairs()
    s1 = load_norm("train", 1)
    s1 = s1[s1["country"] == country].reset_index(drop=True)
    if max_queries and len(s1) > max_queries:
        s1 = s1.sample(n=max_queries, random_state=SEED).reset_index(drop=True)
    print(f"{country}: {len(s1):,} queries | max_df {max_df} | k_rev {k_rev}", flush=True)
    for source in (2, 3):
        docs = load_norm("train", source)
        docs = docs[docs["country"] == country].reset_index(drop=True)
        own = owners(docs, s1, pairs)
        n_true = int((own >= 0).sum())
        t0 = time.time()
        cand = retrieve(s1, docs, k=50, k_rev=k_rev, max_df=max_df)
        took = time.time() - t0
        cand["label"] = own[cand["d"].to_numpy()] == cand["q"].to_numpy()
        print(f"\nS{source}: {len(docs):,} docs, {n_true:,} true pairs, {took:.0f}s, union {len(cand):,} pairs, "
              f"union recall {cand['label'].sum() / n_true:.4f}", flush=True)
        for route in ROUTES:
            rec = [f"@{k}:{cand.loc[cand[f'{route}_rank'] < k, 'label'].sum() / n_true:.3f}" for k in KS]
            line = f"  {route:<6} forward " + " ".join(rec)
            if k_rev:
                line += f" | reverse@{k_rev} {cand.loc[cand[f'{route}_rrank'].notna(), 'label'].sum() / n_true:.3f}"
            print(line, flush=True)
        hit20 = {r: (cand[f"{r}_rank"] < 20).to_numpy() for r in ROUTES}
        uniq = []
        for r in ROUTES:
            others = np.zeros(len(cand), dtype=bool)
            for o in ROUTES:
                if o != r:
                    others |= hit20[o]
            uniq.append(f"{r} {cand.loc[hit20[r] & ~others, 'label'].sum() / n_true:.4f}")
        print("  unique recall @20 (found by no other route): " + " | ".join(uniq), flush=True)
        for k in (3, 5, 10, 20):
            fwd = np.zeros(len(cand), dtype=bool)
            for route in ROUTES:
                fwd |= (cand[f"{route}_rank"] < k).to_numpy()
            line = f"  union forward@{k:<2}: recall {cand.loc[fwd, 'label'].sum() / n_true:.4f}, {fwd.sum() / len(s1):5.1f}/query"
            if k_rev:
                both = fwd.copy()
                for route in ROUTES:
                    both |= cand[f"{route}_rrank"].notna().to_numpy()
                line += f" | + reverse: recall {cand.loc[both, 'label'].sum() / n_true:.4f}, {both.sum() / len(s1):5.1f}/query"
            print(line, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    main(a[0] if a else "India", int(a[1]) if len(a) > 1 else 20000, float(a[2]) if len(a) > 2 else 0.01,
         int(a[3]) if len(a) > 3 else 0)
