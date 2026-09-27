"""Train the candidate filter and the matcher on the train work set; report validation scores.

Run: python -m src.train [TAG]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import ID_BASE, d_key, load_candidates, q_key
from .config import THREADS, WORK_DIR
from .data import load_ground_truth
from .decode import decode_expected_f, decode_threshold, one_owner
from .features import pair_features, retrieval_features
from .lgbutil import predict_chunked
from .metric import oracle_ceiling, scorecard
from .prepare import load_norm

MODEL_DIR = WORK_DIR / "models"
FILTER = {"tau": 0.02, "r": 2, "k_max": 6}
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=THREADS)


def fold_of(q_ids) -> np.ndarray:
    """Deterministic 5-way split by S1 entity (pandas' hash uses a fixed key)."""
    return (pd.util.hash_array(np.asarray(q_ids, dtype=np.int64)) % 5).astype(np.int8)


def truth_sets(q_ids: set[int]) -> dict[int, set]:
    gt = load_ground_truth()
    gt = gt.assign(q=q_key(gt["source1_entity_id"]))
    gt = gt[gt["q"].isin(q_ids)]
    return {int(q): {int(x[1]) * ID_BASE + int(x[3:]) for x in m.split(",")} if m else set()
            for q, m in zip(gt["q"], gt["matched_entity_ids"])}


PAIR_BASE = 4 * ID_BASE  # d keys are < 4e9, so q * PAIR_BASE + d is a unique int64 per pair


def add_labels(cand: pd.DataFrame, truth: dict[int, set]) -> pd.DataFrame:
    true_keys = np.fromiter((q * PAIR_BASE + d for q, ds in truth.items() for d in ds), dtype=np.int64)
    keys = cand["q_id"].to_numpy(np.int64) * PAIR_BASE + cand["d_id"].to_numpy(np.int64)
    cand["label"] = np.isin(keys, true_keys)
    return cand


def filter_features(cand: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in cand.groupby(["country", "source"], sort=False):
        g = g.rename(columns={"q_id": "q", "d_id": "d"})
        parts.append(retrieval_features(g))
    f = pd.concat(parts).sort_index()
    f["source"] = cand["source"].to_numpy(np.int8)
    return f


def apply_filter(cand: pd.DataFrame, p: np.ndarray, tau: float, r: int, k_max: int) -> pd.DataFrame:
    c = cand.assign(p_filter=p)
    c = c[c["p_filter"] >= tau]
    c = c[c.groupby(["source", "d_id"])["p_filter"].rank(ascending=False, method="first") <= r]
    c = c[c.groupby(["source", "q_id"])["p_filter"].rank(ascending=False, method="first") <= k_max]
    return c


def matcher_features(cand: pd.DataFrame, split: str) -> pd.DataFrame:
    s1 = load_norm(split, 1)
    s1.index = q_key(s1["entity_id"])
    parts = []
    for source in (2, 3):
        c = cand[cand["source"] == source]
        if c.empty:
            continue
        docs = load_norm(split, source)
        docs.index = d_key(docs["entity_id"])
        q = s1.loc[c["q_id"].unique()]
        d = docs.loc[c["d_id"].unique()]
        qpos = pd.Series(np.arange(len(q)), index=q.index)
        dpos = pd.Series(np.arange(len(d)), index=d.index)
        q, d = q.reset_index(drop=True), d.reset_index(drop=True)
        cc = c.assign(q=qpos.loc[c["q_id"]].to_numpy(), d=dpos.loc[c["d_id"]].to_numpy())
        f = pair_features(cc, q, d)
        f["p_filter"] = c["p_filter"].to_numpy()
        f.index = c.index
        parts.append(f)
    return pd.concat(parts).loc[cand.index]


def fit(x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, rounds: int = 2000) -> lgb.Booster:
    dtrain, dval = lgb.Dataset(x, y), lgb.Dataset(x_val, y_val)
    return lgb.train(LGB_PARAMS, dtrain, rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def main(tag: str = "work20", reuse_filter: bool = False) -> None:
    t0 = time.time()
    cand = load_candidates("train", tag)
    # Every work-set entity counts, including ones with no candidate at all.
    all_work = set(q_key(pd.read_parquet(WORK_DIR / "cand" / f"work_ids_{tag}.parquet")["entity_id"]).tolist())
    truth = truth_sets(all_work)
    cand = add_labels(cand, truth)
    cand["fold"] = fold_of(cand["q_id"])
    work_arr = np.array(sorted(all_work), dtype=np.int64)
    val_ids = work_arr[fold_of(work_arr) == 4].tolist()
    print(f"{len(cand):,} retrieval pairs for {len(all_work):,} entities ({len(val_ids):,} validation), "
          f"{cand['label'].mean():.3%} positive | {time.time() - t0:.0f}s", flush=True)
    retrieved = cand[cand["label"]].groupby("q_id")["d_id"].agg(set).to_dict()
    print(f"oracle ceiling after retrieval (validation): {oracle_ceiling(retrieved, truth, val_ids):.4f}", flush=True)

    ff = filter_features(cand)
    if reuse_filter and (MODEL_DIR / "filter.txt").exists():
        filt = lgb.Booster(model_file=str(MODEL_DIR / "filter.txt"))
    else:
        tr, va = (cand["fold"] < 4).to_numpy(), (cand["fold"] == 4).to_numpy()
        # The filter only ranks; 8M training rows are plenty and keep training to minutes.
        tr_idx = np.flatnonzero(tr)
        if len(tr_idx) > 8_000_000:
            tr_idx = np.random.default_rng(0).choice(tr_idx, 8_000_000, replace=False)
        filt = fit(ff.iloc[tr_idx], cand["label"].to_numpy()[tr_idx], ff[va], cand["label"][va].to_numpy())
    p_filter = predict_chunked(filt, ff)
    del ff
    (WORK_DIR / "stage").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"q_id": cand["q_id"].to_numpy(), "d_id": cand["d_id"].to_numpy(), "source": cand["source"].to_numpy(),
                  "label": cand["label"].to_numpy(), "fold": cand["fold"].to_numpy(),
                  "p_filter": p_filter.astype(np.float32)}).to_parquet(WORK_DIR / "stage" / f"{tag}_filter_scores.parquet",
                                                                      index=False)
    kept = apply_filter(cand, p_filter, **FILTER)
    per_q = kept[kept["fold"] == 4].groupby("q_id").size()
    kept_true = kept[kept["label"]].groupby("q_id")["d_id"].agg(set).to_dict()
    print(f"filter {FILTER}: {len(kept):,} pairs kept | validation mean {per_q.sum() / len(val_ids):.2f} candidates per "
          f"S1 entity, p95 {per_q.quantile(0.95):.0f} | oracle ceiling {oracle_ceiling(kept_true, truth, val_ids):.4f} "
          f"| {time.time() - t0:.0f}s", flush=True)

    mf = matcher_features(kept, "train")
    print(f"matcher features {mf.shape} | {time.time() - t0:.0f}s", flush=True)
    cache = WORK_DIR / "stage" / f"{tag}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta = kept[["q_id", "d_id", "source", "label", "fold"]].reset_index(drop=True)
    pd.concat([meta, mf.drop(columns=["source"]).reset_index(drop=True)], axis=1).to_parquet(cache, index=False)
    pd.DataFrame({"q_id": work_arr}).to_parquet(cache.with_name(f"{tag}_entities.parquet"), index=False)
    tr, va = (kept["fold"] < 4).to_numpy(), (kept["fold"] == 4).to_numpy()
    model = fit(mf[tr], kept["label"][tr].to_numpy(), mf[va], kept["label"][va].to_numpy())
    kept = kept.assign(p=model.predict(mf, num_threads=THREADS))
    val = kept[kept["fold"] == 4]
    resolved = one_owner(val)
    results = {}
    for name, pairs in (("raw", val), ("one_owner", resolved)):
        best = max(((t, scorecard(decode_threshold(pairs, t), truth, val_ids)["macro_f05"])
                    for t in np.arange(0.2, 0.81, 0.05)), key=lambda x: x[1])
        ef = scorecard(decode_expected_f(pairs), truth, val_ids)
        results[name] = {"best_threshold": round(float(best[0]), 2), "threshold_f05": best[1], "expected_f": ef}
        print(f"{name:<10} best threshold {best[0]:.2f} -> macro F0.5 {best[1]:.4f} | expected-F decoder -> "
              f"{ef['macro_f05']:.4f} (non-singleton {ef['non_singleton_f05']:.4f}, singleton acc "
              f"{ef['singleton_accuracy']:.3f}, share {ef['singleton_share']:.3f})", flush=True)
    imp = pd.Series(model.feature_importance("gain"), index=mf.columns).sort_values(ascending=False)
    print("top matcher features:", ", ".join(f"{k}={v / imp.sum():.1%}" for k, v in imp.head(12).items()))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filt.save_model(str(MODEL_DIR / "filter.txt"))
    model.save_model(str(MODEL_DIR / "matcher.txt"))
    (MODEL_DIR / "validation.json").write_text(json.dumps({"filter": FILTER, "results": results}, indent=2, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "work20", reuse_filter="reuse" in sys.argv[2:])
