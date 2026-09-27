"""Sibling-bridge rescue, applied after decoding (all countries).

Retrieval misses some true matches entirely (the validation ceiling is 0.9948). Many of them share a name or address
key with another record that the matcher already gave, with p >= SEED_P, to the right S1 entity. This step joins
records that no entity claims to those confident "seed" records on the KEYS below, and a LightGBM verifier decides
whether to add each one to the seed's entity. The verifier is trained, and its threshold chosen, on the stage-2
validation entities (India/US, labeled), with the rest of the train population treated as claimed by its true owners.

    python -m src.bridge VAL_PAIRS TEST_SCORES BASE_DIR OUT_DIR
    VAL_PAIRS: work/v3/val_pairs.parquet; TEST_SCORES and BASE_DIR: the output folder of step_predict.
"""
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import d_key, d_str, q_key
from .data import ground_truth_pairs
from .decode import decode_expected_f, one_owner
from .features import string_features
from .prepare import load_norm
from .submission import write_submission
from .train import LGB_PARAMS, PAIR_BASE, truth_sets

SEED_P = 0.995
MAX_KEY_SEEDS = 20  # a name or address shared by more seed records than this is too common to bridge on
# Join keys: sorted word sets of the name and address (reordered copies still link), sorted phonetic name words
# (spelling slips), and house numbers + first street word (typos further along the address).
KEYS = ("name_key", "addr_key", "phon_key", "numst_key")
_NUM = re.compile(r"^\d+[a-z]?$")


def _numst(addr: str) -> str:
    toks = addr.split()
    nums = sorted(t for t in toks if _NUM.match(t))
    words = [t for t in toks if not _NUM.match(t) and len(t) > 2]
    return " ".join(nums + words[:1]) if nums and words else ""


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _norms(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = load_norm(split, 1)
    s1["q_id"] = q_key(s1["entity_id"])
    docs = pd.concat([load_norm(split, s) for s in (2, 3)], ignore_index=True)
    docs["d_id"] = d_key(docs["entity_id"])
    for col, key in (("name_core", "name_key"), ("addr_norm", "addr_key"), ("name_phon", "phon_key")):
        docs[key] = docs[col].map(lambda s: " ".join(sorted(set(s.split()))))
    docs["numst_key"] = docs["addr_norm"].map(_numst)
    return s1, docs


def candidates(pairs: pd.DataFrame, pred: dict, docs: pd.DataFrame, claimed: np.ndarray) -> pd.DataFrame:
    """(d_id, q_id) proposals: an unclaimed record sharing a name or address with a seed record of entity q, and not
    already among q's scored candidates. pairs: every scored (q_id, d_id, p); pred: q_id -> predicted d_ids."""
    pp = pd.DataFrame([(q, d) for q, ds in pred.items() for d in ds], columns=["q_id", "d_id"])
    seeds = pp.merge(pairs[["q_id", "d_id", "p"]], on=["q_id", "d_id"])
    seeds = seeds[seeds["p"] >= SEED_P].rename(columns={"d_id": "s_id", "p": "seed_p"})
    seeds = seeds.join(docs.set_index("d_id")[["country", *KEYS]], on="s_id")
    free = docs.loc[~docs["d_id"].isin(claimed), ["d_id", "country", *KEYS]]
    parts = []
    for key in KEYS:
        s = seeds[seeds[key] != ""]
        s = s[s.groupby(["country", key])["s_id"].transform("size") <= MAX_KEY_SEEDS]
        m = free[free[key] != ""].merge(s[["q_id", "s_id", "seed_p", "country", key]], on=["country", key])
        parts.append(m[["d_id", "q_id", "s_id", "seed_p"]].assign(key=key))
    c = pd.concat(parts, ignore_index=True)
    judged = pairs["q_id"].to_numpy(np.int64) * PAIR_BASE + pairs["d_id"].to_numpy(np.int64)
    c = c[~np.isin(c["q_id"].to_numpy(np.int64) * PAIR_BASE + c["d_id"].to_numpy(np.int64), judged)]
    g = c.groupby(["d_id", "q_id"])
    out = g.agg(s_id=("s_id", "first"), seed_p=("seed_p", "max"), n_seeds=("s_id", "nunique")).reset_index()
    for key in KEYS:
        out[f"by_{key}"] = g["key"].agg(lambda k, key=key: (k == key).any()).to_numpy().astype(np.int8)
    out["n_owners"] = out.groupby("d_id")["q_id"].transform("size").astype(np.int16)
    out["q_n_pred"] = out["q_id"].map({q: len(ds) for q, ds in pred.items()}).astype(np.float32)
    return out


def features(c: pd.DataFrame, s1: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    qpos = pd.Series(np.arange(len(s1)), index=s1["q_id"].to_numpy())
    dpos = pd.Series(np.arange(len(docs)), index=docs["d_id"].to_numpy())
    r = dpos.reindex(c["d_id"].to_numpy()).to_numpy()
    to_s1 = string_features(pd.DataFrame({"q": qpos.reindex(c["q_id"].to_numpy()).to_numpy(), "d": r}), s1, docs)
    to_seed = string_features(pd.DataFrame({"q": dpos.reindex(c["s_id"].to_numpy()).to_numpy(), "d": r}), docs, docs)
    f = pd.concat([to_s1, to_seed.add_prefix("s_")], axis=1)
    for col in ("seed_p", "n_seeds", *(f"by_{k}" for k in KEYS), "n_owners", "q_n_pred"):
        f[col] = c[col].to_numpy(np.float32)
    f["source"] = (c["d_id"].to_numpy() // 10**9).astype(np.int8)
    f["seed_source"] = (c["s_id"].to_numpy() // 10**9).astype(np.int8)
    return f


def _f05(pred: set, truth: set) -> float:
    if not truth:
        return float(not pred)
    tp = len(pred & truth)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(truth) + len(pred))


def _rescues(c: pd.DataFrame, p: np.ndarray, t: float) -> dict:
    """Each record goes to its most probable proposing entity, if that probability reaches t."""
    x = c.assign(p=p)
    x = x[x["p"] >= t].sort_values("p", ascending=False).drop_duplicates("d_id")
    return x.groupby("q_id")["d_id"].agg(set).to_dict()


def validate(val_pairs: str) -> tuple[lgb.Booster, float]:
    """Simulate on validation entities; returns the verifier trained on all of them and the chosen threshold."""
    v = pd.read_parquet(val_pairs).rename(columns={"p2": "p"})
    ids = v["q_id"].unique()
    truth = truth_sets(set(ids.tolist()))
    pred = {q: set(ds) for q, ds in decode_expected_f(one_owner(v)).items()}
    gt = ground_truth_pairs()
    owner = pd.Series(q_key(gt["s1_id"].astype(str)), index=d_key(gt["match_id"].astype(str)))
    s1, docs = _norms("train")
    others = owner[~owner.isin(ids)].index.to_numpy()  # records of non-validation entities count as claimed
    claimed = np.union1d(others, np.fromiter((d for ds in pred.values() for d in ds), dtype=np.int64))
    c = candidates(v[["q_id", "d_id", "p"]], pred, docs, claimed)
    y = (owner.reindex(c["d_id"].to_numpy()).to_numpy() == c["q_id"].to_numpy()).astype(np.int8)
    log(f"validation proposals: {len(c):,} for {c['q_id'].nunique():,} entities | true {y.mean():.3f}")
    x = features(c, s1, docs)
    fold = (pd.util.hash_array(c["q_id"].to_numpy()) % 2).astype(int)
    p = np.zeros(len(c))
    for k in (0, 1):
        tr, te = fold != k, fold == k
        m = lgb.train({**LGB_PARAMS, "num_leaves": 31}, lgb.Dataset(x[tr], y[tr]), 300)
        p[te] = m.predict(x[te])
    base = np.mean([_f05(pred.get(q, set()), truth[q]) for q in ids])
    best_t, best = 1.01, base
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        add = _rescues(c, p, t)
        score = np.mean([_f05(pred.get(q, set()) | add.get(q, set()), truth[q]) for q in ids])
        n = sum(len(s) for s in add.values())
        prec = np.mean([d in truth[q] for q, s in add.items() for d in s]) if n else float("nan")
        log(f"  t={t:.2f}: {n:,} records added, precision {prec:.3f} | macro F0.5 {base:.5f} -> {score:.5f}")
        if score > best:
            best_t, best = t, score
    log(f"chosen threshold {best_t} (validation gain {best - base:+.5f})")
    return lgb.train({**LGB_PARAMS, "num_leaves": 31}, lgb.Dataset(x, y), 300), best_t


def _read_lists(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s1: [x for x in ids.split(",") if x] for s1, ids in zip(df.iloc[:, 0], df.iloc[:, 1])}


def apply(model: lgb.Booster, t: float, test_scores: str, base_dir: str, out_dir: str) -> None:
    pairs = pd.read_parquet(test_scores, columns=["q_id", "d_id", "p"])
    matches = _read_lists(Path(base_dir) / "matching_results.tsv")
    cands = _read_lists(Path(base_dir) / "candidate_pairs.tsv")
    pred = {int(s[3:]): set(d_key(pd.Series(ds, dtype=str)).tolist()) for s, ds in matches.items() if ds}
    s1, docs = _norms("test")
    claimed = np.fromiter((d for ds in pred.values() for d in ds), dtype=np.int64)
    c = candidates(pairs, pred, docs, claimed)
    add = _rescues(c, model.predict(features(c, s1, docs)), t) if len(c) else {}
    country = s1.set_index("q_id")["country"]
    added = pd.Series({q: len(s) for q, s in add.items()}, dtype=float)
    log(f"test: {len(c):,} proposals, {int(added.sum()):,} records added to {len(added):,} entities; by country: "
        f"{added.groupby(country.reindex(added.index)).sum().to_dict()}")
    for q, s in add.items():
        key = f"S1-{q}"
        extra = d_str(sorted(s))
        matches[key] = matches[key] + extra
        cands[key] = cands[key] + extra
    write_submission(Path(out_dir), list(cands), matches, cands)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    verifier, threshold = validate(sys.argv[1])
    if threshold <= 1:
        apply(verifier, threshold, sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        log("no threshold improves validation; nothing written")
