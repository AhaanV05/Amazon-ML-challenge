"""Stage-1 out-of-fold scores, stage-2 context features, validation, and final models.

Run after `python -m src.train TAG` has written work/stage/TAG.parquet:  python -m src.train2 TAG
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import d_key
from .ce_data import CE_DIR, attach_ce
from .config import THREADS, WORK_DIR
from .context import context_features
from .decode import decode_expected_f, decode_threshold, one_owner
from .metric import scorecard
from .prepare import load_norm
from .train import LGB_PARAMS, MODEL_DIR, truth_sets

META = ["q_id", "d_id", "label", "fold"]  # everything else, including `source`, is a model feature


def fit(x, y, x_val, y_val, rounds=3000):
    return lgb.train(LGB_PARAMS, lgb.Dataset(x, y), rounds, valid_sets=[lgb.Dataset(x_val, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def fit_fixed(x, y, rounds):
    return lgb.train(LGB_PARAMS, lgb.Dataset(x, y), rounds)


def docs_frame(split: str, d_ids: np.ndarray) -> pd.DataFrame:
    parts = []
    wanted = set(d_ids.tolist())
    for source in (2, 3):
        d = load_norm(split, source)
        d.index = d_key(d["entity_id"])
        parts.append(d[d.index.isin(wanted)])
    return pd.concat(parts)


def evaluate(pairs: pd.DataFrame, truth, ids, label: str) -> dict:
    best = max(((t, scorecard(decode_threshold(pairs, t), truth, ids)["macro_f05"]) for t in np.arange(0.3, 0.86, 0.05)),
               key=lambda x: x[1])
    ef = scorecard(decode_expected_f(pairs), truth, ids)
    print(f"  {label:<22} threshold {best[0]:.2f} -> {best[1]:.4f} | expected-F -> {ef['macro_f05']:.4f} "
          f"(non-singleton {ef['non_singleton_f05']:.4f}, singleton acc {ef['singleton_accuracy']:.3f})", flush=True)
    return {"best_threshold": round(float(best[0]), 2), "threshold_f05": best[1], "expected_f": ef}


def main(tag: str = "work20") -> None:
    t0 = time.time()
    df = pd.read_parquet(WORK_DIR / "stage" / f"{tag}.parquet")
    entities = pd.read_parquet(WORK_DIR / "stage" / f"{tag}_entities.parquet")["q_id"].to_numpy()
    use_ce = (CE_DIR / "work_scores.parquet").exists() and "noce" not in sys.argv[2:]
    if use_ce:
        # Trained on entities disjoint from this work set, so these scores are out-of-sample here.
        df["ce_logit"] = attach_ce(df, "work")
        print(f"cross-encoder feature attached ({df['ce_logit'].notna().mean():.1%} of pairs)", flush=True)
    feats = [c for c in df.columns if c not in META]
    x, y, fold = df[feats], df["label"].to_numpy(), df["fold"].to_numpy()
    from .train import fold_of
    val_ids = entities[fold_of(entities) == 4].tolist()
    truth = truth_sets(set(entities.tolist()))

    p1 = np.zeros(len(df))
    rounds = []
    for k in range(5):
        tr, te = fold != k, fold == k
        m = fit(x[tr], y[tr], x[te], y[te])
        p1[te] = m.predict(x[te], num_threads=THREADS)
        rounds.append(m.best_iteration)
        print(f"stage 1 fold {k}: {m.best_iteration} rounds | {time.time() - t0:.0f}s", flush=True)

    docs = docs_frame("train", df["d_id"].unique())
    ctx = context_features(df, p1, docs)
    x2 = pd.concat([x.reset_index(drop=True), ctx], axis=1)
    print(f"context features {ctx.shape} | {time.time() - t0:.0f}s", flush=True)
    tr, va = fold < 4, fold == 4
    m2 = fit(x2[tr], y[tr], x2[va], y[va])
    p2 = m2.predict(x2[va], num_threads=THREADS)

    val = df.loc[va, ["q_id", "d_id"]].reset_index(drop=True)
    print("validation (fold 4):")
    results = {
        "stage1": evaluate(one_owner(val.assign(p=p1[va])), truth, val_ids, "stage 1 + one-owner"),
        "stage2": evaluate(one_owner(val.assign(p=p2)), truth, val_ids, "stage 2 + one-owner"),
    }
    imp = pd.Series(m2.feature_importance("gain"), index=x2.columns).sort_values(ascending=False)
    print("top stage-2 features:", ", ".join(f"{k}={v / imp.sum():.1%}" for k, v in imp.head(12).items()))

    n1 = int(np.mean(rounds) * 1.1)
    n2 = int(m2.best_iteration * 1.1)
    final1 = fit_fixed(x, y, n1)
    final2 = fit_fixed(x2, y, n2)
    final1.save_model(str(MODEL_DIR / "stage1.txt"))
    final2.save_model(str(MODEL_DIR / "stage2.txt"))
    (MODEL_DIR / "validation2.json").write_text(json.dumps({"rounds": [n1, n2], "cross_encoder": use_ce,
                                                            "results": results}, indent=2, default=str))
    print(f"final models saved ({n1}, {n2} rounds) | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "work20")
