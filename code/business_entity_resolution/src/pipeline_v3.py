"""Population-consistent pipeline, sized for a large machine (48 cores, 176 GB RAM, one big GPU).

Every train S1 entity goes through retrieval, the candidate filter, cross-encoder scoring and feature computation,
exactly like test. Features that count competition for a record ("how many entities retrieved it", "the best
score any other entity has for it") therefore have the same distribution at train and test time. Models train on
a sample of entities that is disjoint from the cross-encoder's own training entities.

    python -m src.pipeline_v3 <step>
    steps, in order: prepare, candidates, filter, ce_export, ce_score, features, stage1, stage2, predict

Environment: BER_WORK_DIR, BER_OUTPUT_DIR, BER_THREADS, BER_K (retrieval top-k, default 20),
BER_CAND_BUDGET (max mean candidates per S1 entity, default 16), BER_MODEL_FRAC (default 0.3),
BER_CE_MODEL (cross-encoder directory), BER_CE_PYTHON (python with torch), BER_SMOKE=1 (tiny end-to-end run).
"""
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import ID_BASE, cand_dir, d_str, generate, q_key
from .ce_data import record_texts
from .config import OUTPUT_DIR, SEED, WORK_DIR
from .context import context_features
from .data import load_table
from .decode import decode_expected_f, decode_threshold, one_owner
from .lgbutil import predict_chunked, predict_frames
from .prepare import load_norm
from .retrieve import ROUTE_NAMES
from .submission import write_submission
from .train import LGB_PARAMS, PAIR_BASE, add_labels, filter_features, fit, fold_of, matcher_features, truth_sets
from .train2 import docs_frame, evaluate

SMOKE = os.environ.get("BER_SMOKE") == "1"
TAG = "smoke" if SMOKE else "all"
V3 = WORK_DIR / ("v3_smoke" if SMOKE else "v3")
K = int(os.environ.get("BER_K", 20))
CAND_BUDGET = float(os.environ.get("BER_CAND_BUDGET", 16))
MODEL_FRAC = float(os.environ.get("BER_MODEL_FRAC", 0.3))
# v6: pipeline_v4's two-fold rerankers give every train entity an out-of-sample score, so the matcher (stages 1-2)
# can learn from all train entities instead of the MODEL_FRAC sample. Rerankers still split entities by roles().
STAGE_ALL = os.environ.get("BER_STAGE_ALL") == "1"
FILTER_SAMPLE = 0.25
CE_MODEL = Path(os.environ.get("BER_CE_MODEL", WORK_DIR / "ce_model"))
CE_LARGE_MODEL = Path(os.environ.get("BER_CE_LARGE_MODEL", WORK_DIR / "ce_large_model"))
LARGE_BAND = (0.02, 0.98)  # the large cross-encoder rescores only pairs the base model is unsure about
CE_PYTHON = os.environ.get("BER_CE_PYTHON", sys.executable)
META = ["q_id", "d_id", "label", "country"]
ROUTE_COLS = [f"{r}_{s}" for r in ROUTE_NAMES for s in ("score", "rank")]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def smoke_subset(split: str) -> set[str] | None:
    if not SMOKE:
        return None
    s1 = load_norm(split, 1)[["entity_id", "country"]]
    return set(s1.groupby("country").sample(n=1500, random_state=SEED)["entity_id"])


def roles() -> pd.DataFrame:
    """Every train S1 entity: cross-encoder training entity, model entity, or population only."""
    path = V3 / "roles.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s1 = load_norm("train", 1)[["entity_id", "country"]]
    sub = smoke_subset("train")
    if sub is not None:
        s1 = s1[s1["entity_id"].isin(sub)]
    q = q_key(s1["entity_id"])
    ce = set(q_key(pd.read_parquet(WORK_DIR / "cand" / "work_ids_ce10.parquet")["entity_id"]).tolist())
    is_ce = np.isin(q, np.fromiter(ce, dtype=np.int64))
    frac = 0.8 if SMOKE else MODEL_FRAC
    is_model = ~is_ce & (np.random.default_rng(SEED).random(len(q)) < frac)
    out = pd.DataFrame({"q_id": q, "country": s1["country"].to_numpy(), "is_ce": is_ce, "is_model": is_model,
                        "fold": fold_of(q)})
    V3.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def _files(split: str):
    for path in sorted(cand_dir(split, TAG).glob("*.parquet")):
        cand = pd.read_parquet(path)
        if cand.empty:
            continue
        country, source = path.stem.rsplit("_s", 1)
        yield path.stem, cand.assign(country=country, source=np.int8(source))


def _ranks(cand: pd.DataFrame) -> pd.DataFrame:
    """Rank of each pair by filter score among all pairs of its record and of its entity (whole population)."""
    cand["rank_d"] = cand.groupby(["source", "d_id"])["p_filter"].rank(ascending=False, method="first").astype(np.float32)
    cand["rank_q"] = cand.groupby(["source", "q_id"])["p_filter"].rank(ascending=False, method="first").astype(np.float32)
    return cand


def _select(cand: pd.DataFrame, tau: float, r: float, k_max: float) -> pd.DataFrame:
    keep = (cand["p_filter"] >= tau) & (cand["rank_d"] <= r) & (cand["rank_q"] <= k_max)
    return cand[keep].drop(columns=["rank_d", "rank_q"])


def _attach(frame: pd.DataFrame, scores_path: Path) -> np.ndarray:
    s = pd.read_parquet(scores_path)
    keys = (s["q_rec"].to_numpy(np.int64) - ID_BASE) * PAIR_BASE + s["d_rec"].to_numpy(np.int64)
    lookup = pd.Series(s["ce_logit"].to_numpy(np.float32), index=keys)
    want = frame["q_id"].to_numpy(np.int64) * PAIR_BASE + frame["d_id"].to_numpy(np.int64)
    return lookup.reindex(want).to_numpy(np.float32)


# ---------------------------------------------------------------------------------------------------------------
def step_prepare() -> None:
    from .prepare import main as prepare_main
    prepare_main()


def step_candidates() -> None:
    t0 = time.time()
    for split in ("train", "test"):
        generate(split, TAG, smoke_subset(split), K)
    log(f"candidates done in {time.time() - t0:.0f}s")


def step_filter() -> None:
    t0 = time.time()
    r = roles()
    truth = truth_sets(set(r["q_id"].tolist()))
    is_model = pd.Series(r["is_model"].to_numpy(), index=r["q_id"].to_numpy())
    fold = pd.Series(r["fold"].to_numpy(), index=r["q_id"].to_numpy())
    rng = np.random.default_rng(SEED)
    store, tr_x, tr_y, va_x, va_y = {}, [], [], [], []
    for stem, cand in _files("train"):
        cand = add_labels(cand, truth)
        ff = filter_features(cand)
        q = cand["q_id"].to_numpy()
        m, f = is_model.reindex(q).fillna(False).to_numpy(bool), fold.reindex(q).fillna(-1).to_numpy()
        for idx, xs, ys in ((np.flatnonzero(m & (f < 4)), tr_x, tr_y), (np.flatnonzero(m & (f == 4)), va_x, va_y)):
            take = idx if SMOKE else rng.choice(idx, size=int(len(idx) * FILTER_SAMPLE), replace=False)
            xs.append(ff.iloc[take])
            ys.append(cand["label"].to_numpy()[take])
        store[stem] = (cand, ff)
        log(f"filter features train/{stem}: {len(cand):,} pairs | {time.time() - t0:.0f}s")
    filt = fit(pd.concat(tr_x), np.concatenate(tr_y), pd.concat(va_x), np.concatenate(va_y))
    filt.save_model(str(V3 / "filter.txt"))
    del tr_x, tr_y, va_x, va_y
    log(f"filter model: {filt.best_iteration} rounds | {time.time() - t0:.0f}s")

    val_parts = []
    for stem, (cand, ff) in list(store.items()):
        cand["p_filter"] = predict_chunked(filt, ff).astype(np.float32)
        cand = _ranks(cand)
        q = cand["q_id"].to_numpy()
        vm = is_model.reindex(q).fillna(False).to_numpy(bool) & (fold.reindex(q).fillna(-1).to_numpy() == 4)
        val_parts.append(cand.loc[vm, ["q_id", "label", "p_filter", "rank_d", "rank_q"]])
        store[stem] = cand
    val = pd.concat(val_parts, ignore_index=True)
    val_ids = r.loc[r["is_model"] & (r["fold"] == 4), "q_id"].to_numpy()
    m_true = np.array([len(truth[q]) for q in val_ids], dtype=np.float64)
    rows = []
    for tau, rr, kk in itertools.product((0.0005, 0.001, 0.002, 0.005, 0.01, 0.02), (2, 3, 5, 99), (6, 8, 10, 12, 15)):
        kept = val[(val["p_filter"] >= tau) & (val["rank_d"] <= rr) & (val["rank_q"] <= kk)]
        n_c = kept.groupby("q_id").size().reindex(val_ids, fill_value=0).to_numpy()
        c = kept[kept["label"]].groupby("q_id").size().reindex(val_ids, fill_value=0).to_numpy()
        oracle = np.where(m_true == 0, 1.0, 1.25 * c / np.maximum(c + 0.25 * m_true, 1e-9)).mean()
        rows.append({"tau": tau, "r": rr, "k_max": kk, "mean_cands": n_c.mean(), "p95": np.percentile(n_c, 95),
                     "oracle": oracle})
    sweep = pd.DataFrame(rows).sort_values("mean_cands")
    sweep.to_csv(V3 / "filter_sweep.csv", index=False)
    ok = sweep[sweep["mean_cands"] <= CAND_BUDGET]
    best = (ok if len(ok) else sweep.head(1)).sort_values(["oracle", "mean_cands"], ascending=[False, True]).iloc[0]
    op = {"tau": float(best["tau"]), "r": float(best["r"]), "k_max": float(best["k_max"])}
    (V3 / "filter_op.json").write_text(json.dumps({**op, "mean_cands": best["mean_cands"], "oracle": best["oracle"]}))
    all_retrieved = val.groupby("q_id").size().reindex(val_ids, fill_value=0).mean()
    log(f"validation: all retrieved {all_retrieved:.1f}/entity | chosen {op} -> {best['mean_cands']:.2f}/entity, "
        f"p95 {best['p95']:.0f}, oracle ceiling {best['oracle']:.4f}")
    print(sweep.sort_values("oracle", ascending=False).head(12).to_string(index=False), flush=True)

    keep_cols = ["q_id", "d_id", "country", "source"] + ROUTE_COLS + ["p_filter"]
    kept_train = pd.concat([_select(c, **op)[keep_cols + ["label"]] for c in store.values()], ignore_index=True)
    del store
    kept_train["country"] = kept_train["country"].astype("category")
    kept_train.to_parquet(V3 / "train_kept.parquet", index=False)
    log(f"train kept {len(kept_train):,} pairs ({len(kept_train) / len(r):.2f}/entity) | {time.time() - t0:.0f}s")

    test_parts = []
    for stem, cand in _files("test"):
        cand["p_filter"] = predict_chunked(filt, filter_features(cand)).astype(np.float32)
        test_parts.append(_select(_ranks(cand), **op)[keep_cols])
        log(f"filter test/{stem}: {len(cand):,} -> {len(test_parts[-1]):,} | {time.time() - t0:.0f}s")
    kept_test = pd.concat(test_parts, ignore_index=True)
    kept_test["country"] = kept_test["country"].astype("category")
    kept_test.to_parquet(V3 / "test_kept.parquet", index=False)
    n_test = load_table("test", "source1").num_rows if not SMOKE else len(smoke_subset("test"))
    log(f"test kept {len(kept_test):,} pairs ({len(kept_test) / n_test:.2f}/entity) | {time.time() - t0:.0f}s")


def step_ce_export() -> None:
    t0 = time.time()
    ce_dir = V3 / "ce"
    ce_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        kept = pd.read_parquet(V3 / f"{split}_kept.parquet", columns=["q_id", "d_id"])
        pairs = pd.DataFrame({"q_rec": kept["q_id"].to_numpy(np.int64) + ID_BASE, "d_rec": kept["d_id"].to_numpy(np.int64)})
        pairs.to_parquet(ce_dir / f"{split}_pairs.parquet", index=False)
        record_texts(split, np.concatenate([pairs["q_rec"], pairs["d_rec"]])).to_parquet(
            ce_dir / f"records_{split}.parquet", index=False)
        log(f"ce export {split}: {len(pairs):,} pairs | {time.time() - t0:.0f}s")


def step_ce_score() -> None:
    ce_dir = V3 / "ce"
    for split in ("train", "test"):
        if SMOKE:
            pairs = pd.read_parquet(ce_dir / f"{split}_pairs.parquet")
            pairs.assign(ce_logit=np.random.default_rng(0).normal(size=len(pairs)).astype(np.float32)).to_parquet(
                ce_dir / f"{split}_scores.parquet", index=False)
            continue
        subprocess.run([CE_PYTHON, str(Path(__file__).with_name("cross_encoder.py")), "score", "--data", str(ce_dir),
                        "--out", str(CE_MODEL), "--pairs", f"{split}_pairs.parquet", "--records",
                        f"records_{split}.parquet", "--workers", "24", "--infer-batch", "4096"], check=True)
    log("CE DONE")


def step_ce_large() -> None:
    """Rescore the uncertain band with the larger cross-encoder, if it was trained. Optional."""
    ce_dir = V3 / "ce"
    if not (CE_LARGE_MODEL / "config.json").exists():
        log(f"large cross-encoder not found at {CE_LARGE_MODEL}; skipping")
        return
    for split in ("train", "test"):
        s = pd.read_parquet(ce_dir / f"{split}_scores.parquet")
        p = 1 / (1 + np.exp(-s["ce_logit"].to_numpy(np.float64)))
        band = s.loc[(p >= LARGE_BAND[0]) & (p <= LARGE_BAND[1]), ["q_rec", "d_rec"]]
        band.to_parquet(ce_dir / f"{split}_large_pairs.parquet", index=False)
        log(f"large cross-encoder {split}: {len(band):,} uncertain pairs ({len(band) / len(s):.1%})")
        subprocess.run([CE_PYTHON, str(Path(__file__).with_name("cross_encoder.py")), "score", "--data", str(ce_dir),
                        "--out", str(CE_LARGE_MODEL), "--pairs", f"{split}_large_pairs.parquet", "--records",
                        f"records_{split}.parquet", "--workers", "24", "--infer-batch", "1024"], check=True)
    log("CE LARGE DONE")


def step_features() -> None:
    t0 = time.time()
    for split in ("train", "test"):
        kept = pd.read_parquet(V3 / f"{split}_kept.parquet")
        mf = matcher_features(kept, split)
        meta = ["q_id", "d_id", "country", "source"] + (["label"] if split == "train" else [])
        out = pd.concat([kept[meta].reset_index(drop=True), mf.drop(columns=["source"]).reset_index(drop=True)], axis=1)
        out.to_parquet(V3 / f"features_{split}.parquet", index=False)
        log(f"features {split}: {out.shape} | {time.time() - t0:.0f}s")


def _load_features(split: str) -> pd.DataFrame:
    df = pd.read_parquet(V3 / f"features_{split}.parquet")
    df["ce_logit"] = _attach(df, V3 / "ce" / f"{split}_scores.parquet")
    # Only when both splits have it, so train and test always share one feature set. NaN outside the band.
    if all((V3 / "ce" / f"{s}_large_scores.parquet").exists() for s in ("train", "test")):
        df["ce_large_logit"] = _attach(df, V3 / "ce" / f"{split}_large_scores.parquet")
    if all((V3 / "ce" / f"{s}_llm_scores.parquet").exists() for s in ("train", "test")):  # v5 LLM reranker
        df["llm_logit"] = _attach(df, V3 / "ce" / f"{split}_llm_scores.parquet")
    return df


def _stage_roles() -> pd.DataFrame:
    """Roles as the matcher sees them: with BER_STAGE_ALL=1 every train entity is a model entity."""
    r = roles()
    return r.assign(is_model=True) if STAGE_ALL else r


def _model_masks(df: pd.DataFrame, r: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = df["q_id"].to_numpy()
    is_model = pd.Series(r["is_model"].to_numpy(), index=r["q_id"].to_numpy()).reindex(q).fillna(False).to_numpy(bool)
    fold = pd.Series(r["fold"].to_numpy(), index=r["q_id"].to_numpy()).reindex(q).fillna(-1).to_numpy()
    return is_model, fold


def step_stage1() -> None:
    t0 = time.time()
    r = _stage_roles()
    tr = _load_features("train")
    feats = [c for c in tr.columns if c not in META]
    y = tr["label"].to_numpy()
    model_rows, fold = _model_masks(tr, r)
    log(f"stage 1: {len(tr):,} train pairs, {model_rows.sum():,} from model entities, {len(feats)} features, "
        f"cross-encoder coverage {np.isfinite(tr['ce_logit']).mean():.1%}")
    p1 = np.full(len(tr), np.nan)
    rounds = []
    for k in range(5):
        a, b = model_rows & (fold != k), model_rows & (fold == k)
        m = fit(tr.loc[a, feats], y[a], tr.loc[b, feats], y[b])
        p1[b] = predict_chunked(m, tr.loc[b, feats])
        rounds.append(m.best_iteration)
        log(f"stage 1 fold {k}: {m.best_iteration} rounds | {time.time() - t0:.0f}s")
    full = lgb.train(LGB_PARAMS, lgb.Dataset(tr.loc[model_rows, feats], y[model_rows]), int(np.mean(rounds) * 1.1) + 1)
    full.save_model(str(V3 / "stage1.txt"))
    if (~model_rows).any():  # none with BER_STAGE_ALL=1
        p1[~model_rows] = predict_chunked(full, tr.loc[~model_rows, feats])
    np.save(V3 / "p1_train.npy", p1)
    te = _load_features("test")
    np.save(V3 / "p1_test.npy", predict_chunked(full, te[feats]))
    log(f"stage 1 done | {time.time() - t0:.0f}s")


def step_stage2() -> None:
    t0 = time.time()
    r = _stage_roles()
    tr = _load_features("train")
    feats = [c for c in tr.columns if c not in META]
    y = tr["label"].to_numpy()
    p1 = np.load(V3 / "p1_train.npy")
    ctx = context_features(tr[["q_id", "d_id", "source"]], p1, docs_frame("train", tr["d_id"].unique()))
    log(f"train context features {ctx.shape} | {time.time() - t0:.0f}s")
    model_rows, fold = _model_masks(tr, r)
    a, b = model_rows & (fold < 4), model_rows & (fold == 4)
    x = pd.concat([tr[feats].reset_index(drop=True), ctx], axis=1)
    m2 = fit(x[a], y[a], x[b], y[b])
    p_eval = p1.copy()
    p_eval[b] = m2.predict(x[b].to_numpy(np.float32))
    log(f"stage 2 validation model: {m2.best_iteration} rounds | {time.time() - t0:.0f}s")
    tr.loc[b, ["q_id", "d_id", "label"]].assign(p1=p1[b].astype(np.float32), p2=p_eval[b].astype(np.float32)).to_parquet(
        V3 / "val_pairs.parquet", index=False)  # for error analysis

    val_ids = r.loc[r["is_model"] & (r["fold"] == 4), "q_id"].to_numpy()
    truth = truth_sets(set(val_ids.tolist()))
    base = tr[["q_id", "d_id", "country"]].reset_index(drop=True)
    results = {}
    for name, p in (("stage1", p1), ("stage2", p_eval)):
        resolved = one_owner(base.assign(p=p))  # competition resolved over the whole train population
        results[name] = evaluate(resolved[resolved["q_id"].isin(val_ids)], truth, val_ids.tolist(), f"{name} + one-owner")
    resolved = one_owner(base.assign(p=p_eval))
    resolved = resolved[resolved["q_id"].isin(val_ids)]
    by_country = r.loc[r["q_id"].isin(val_ids)].groupby("country")["q_id"].agg(list)
    for country, ids in by_country.items():
        evaluate(resolved[resolved["q_id"].isin(ids)], {q: truth[q] for q in ids}, ids, f"  stage 2 {country}")

    final = lgb.train(LGB_PARAMS, lgb.Dataset(x[model_rows], y[model_rows]), int(m2.best_iteration * 1.1) + 1)
    final.save_model(str(V3 / "stage2.txt"))
    del x
    te = _load_features("test")
    p1_te = np.load(V3 / "p1_test.npy")
    ctx_te = context_features(te[["q_id", "d_id", "source"]], p1_te, docs_frame("test", te["d_id"].unique()))
    np.save(V3 / "p2_test.npy", predict_frames(final, [te[feats], ctx_te]))
    (V3 / "validation.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"stage 2 done | {time.time() - t0:.0f}s")


def step_predict() -> None:
    t0 = time.time()
    te = pd.read_parquet(V3 / "test_kept.parquet", columns=["q_id", "d_id", "country"])
    te["p"] = np.load(V3 / "p2_test.npy")
    chosen = json.loads((V3 / "validation.json").read_text())["stage2"]
    resolved = one_owner(te)
    if chosen["expected_f"]["macro_f05"] > chosen["threshold_f05"]:
        matches, how = decode_expected_f(resolved), "expected-F0.5"
    else:
        matches, how = decode_threshold(resolved, chosen["best_threshold"]), f"threshold {chosen['best_threshold']}"
    s1_ids = load_table("test", "source1")["entity_id"].to_pylist()
    cands = {f"S1-{q}": d_str(ds) for q, ds in te.groupby("q_id")["d_id"].agg(list).items()}
    preds = {f"S1-{q}": d_str(ds) for q, ds in matches.items()}
    write_submission(OUTPUT_DIR, s1_ids, preds, cands)
    te[["q_id", "d_id", "country", "p"]].to_parquet(OUTPUT_DIR / "test_scores.parquet", index=False)  # for re-decoding
    n_c, n_m = sum(map(len, cands.values())), sum(map(len, preds.values()))
    log(f"decoder {how} | candidates/S1 {n_c / len(s1_ids):.2f} | matches/S1 {n_m / len(s1_ids):.2f} | "
        f"empty {1 - len(preds) / len(s1_ids):.1%} | written to {OUTPUT_DIR} | {time.time() - t0:.0f}s")


STEPS = {"prepare": step_prepare, "candidates": step_candidates, "filter": step_filter, "ce_export": step_ce_export,
         "ce_score": step_ce_score, "ce_large": step_ce_large, "features": step_features, "stage1": step_stage1,
         "stage2": step_stage2, "predict": step_predict}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    V3.mkdir(parents=True, exist_ok=True)
    for step in sys.argv[1:]:
        log(f"=== {step} ===")
        STEPS[step]()
