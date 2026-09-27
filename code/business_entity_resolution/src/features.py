"""Pair features for candidate (S1 record, S2/S3 record) pairs."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .retrieve import ROUTE_NAMES

WORKERS = -1


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32)


def _set_stats(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jac, shared, conflict = [], [], []
    for x, y in zip(a, b):
        sx, sy = set(x.split()), set(y.split())
        if sx and sy:
            inter = len(sx & sy)
            jac.append(inter / len(sx | sy))
            shared.append(inter)
            conflict.append(inter == 0)
        else:
            jac.append(np.nan)
            shared.append(0)
            conflict.append(False)
    return np.array(jac, np.float32), np.array(shared, np.int8), np.array(conflict, np.int8)


def retrieval_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Cheap features available right after retrieval; the candidate filter uses only these."""
    f = pd.DataFrame(index=cand.index)
    hits = np.zeros(len(cand), np.int8)
    for route in ROUTE_NAMES:
        score, rank = cand.get(f"{route}_score"), cand.get(f"{route}_rank")
        if score is None:
            continue
        f[f"{route}_score"] = score.astype(np.float32)
        f[f"{route}_rank"] = rank.astype(np.float32)
        hits += score.notna().to_numpy()
    f["n_routes"] = hits
    best = f[[c for c in f.columns if c.endswith("_score")]].max(axis=1)
    f["best_score"] = best
    grp = cand["q"]
    f["q_best_score"] = best.groupby(grp).transform("max")
    f["gap_to_q_best"] = f["q_best_score"] - best
    f["rank_in_q"] = best.groupby(grp).rank(ascending=False, method="min").astype(np.float32)
    f["q_n_cands"] = grp.map(grp.value_counts()).astype(np.float32)
    dgrp = cand["d"]
    f["d_n_queries"] = dgrp.map(dgrp.value_counts()).astype(np.float32)
    f["d_best_score"] = best.groupby(dgrp).transform("max")
    f["gap_to_d_best"] = f["d_best_score"] - best
    f["rank_in_d"] = best.groupby(dgrp).rank(ascending=False, method="min").astype(np.float32)
    return f.astype(np.float32)


def pair_features(cand: pd.DataFrame, q: pd.DataFrame, d: pd.DataFrame, chunk: int = 1_000_000) -> pd.DataFrame:
    """cand has integer columns q, d indexing rows of q and d (normalized frames).

    Context features see the whole candidate set; string features are row-wise, so they run in chunks to
    bound the memory of the Python string lists.
    """
    f = retrieval_features(cand).reset_index(drop=True)
    parts = [string_features(cand.iloc[i:i + chunk], q, d) for i in range(0, len(cand), chunk)]
    f = pd.concat([f, pd.concat(parts, ignore_index=True)], axis=1)
    f["source"] = cand["source"].to_numpy(np.int8)
    return f


def string_features(cand: pd.DataFrame, q: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    a = q.iloc[cand["q"].to_numpy()].reset_index(drop=True)
    b = d.iloc[cand["d"].to_numpy()].reset_index(drop=True)
    f = pd.DataFrame(index=pd.RangeIndex(len(cand)))
    an, bn = a["name_core"].tolist(), b["name_core"].tolist()
    ap, bp = a["name_phon"].tolist(), b["name_phon"].tolist()
    aa, ba = a["addr_norm"].tolist(), b["addr_norm"].tolist()
    f["name_ratio"] = _cp(an, bn, fuzz.ratio)
    f["name_token_sort"] = _cp(an, bn, fuzz.token_sort_ratio)
    f["name_token_set"] = _cp(an, bn, fuzz.token_set_ratio)
    f["name_partial"] = _cp(an, bn, fuzz.partial_ratio)
    f["name_jw"] = _cp(an, bn, JaroWinkler.normalized_similarity)
    f["phon_ratio"] = _cp(ap, bp, fuzz.ratio)
    f["phon_token_set"] = _cp(ap, bp, fuzz.token_set_ratio)
    f["addr_ratio"] = _cp(aa, ba, fuzz.ratio)
    f["addr_token_sort"] = _cp(aa, ba, fuzz.token_sort_ratio)
    f["addr_token_set"] = _cp(aa, ba, fuzz.token_set_ratio)
    f["addr_partial"] = _cp(aa, ba, fuzz.partial_ratio)
    f["num_jaccard"], f["num_shared"], f["num_conflict"] = _set_stats(a["addr_nums"], b["addr_nums"])
    f["name_jaccard"], _, _ = _set_stats(a["name_core"], b["name_core"])
    f["addr_jaccard"], _, _ = _set_stats(a["addr_norm"], b["addr_norm"])
    la, lb = a["name_legal"].to_numpy(), b["name_legal"].to_numpy()
    f["legal_state"] = np.where((la == "") | (lb == ""), 0, np.where(la == lb, 1, -1)).astype(np.int8)
    pa_, pb_ = a["postcode"].to_numpy(), b["postcode"].to_numpy()
    f["postcode_state"] = np.where((pa_ == "") | (pb_ == ""), 0, np.where(pa_ == pb_, 1, -1)).astype(np.int8)
    f["name_len_a"] = a["name_core"].str.len().to_numpy(np.float32)
    f["name_len_b"] = b["name_core"].str.len().to_numpy(np.float32)
    f["addr_len_a"] = a["addr_norm"].str.len().to_numpy(np.float32)
    f["addr_len_b"] = b["addr_norm"].str.len().to_numpy(np.float32)
    f["name_ntok_b"] = b["name_core"].str.count(" ").to_numpy(np.float32) + (b["name_core"] != "").to_numpy()
    first_a = a["name_core"].str.split(" ").str[0]
    first_b = b["name_core"].str.split(" ").str[0]
    f["first_token_equal"] = (first_a == first_b).to_numpy(np.int8)
    f["b_domain"] = b["name_domain"].to_numpy(np.int8)
    f["b_name_indic"] = b["name_indic"].to_numpy(np.int8)
    f["b_addr_indic"] = b["addr_indic"].to_numpy(np.int8)
    f["b_addr_empty"] = (b["addr_norm"] == "").to_numpy(np.int8)
    return f
