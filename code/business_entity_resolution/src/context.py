"""Stage-2 context features built from stage-1 pair probabilities over the whole candidate graph."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def _anchor(df: pd.DataFrame, mask: np.ndarray | None = None) -> np.ndarray:
    """For each row, the d_id of the highest-p1 other candidate of the same q (restricted by mask), or -1."""
    sub = df if mask is None else df[mask]
    order = sub.sort_values(["q_id", "p1"], ascending=[True, False])
    first = order.groupby("q_id")["d_id"].first()
    second = order.groupby("q_id")["d_id"].nth(1)
    second = pd.Series(second.to_numpy(), index=order.loc[second.index, "q_id"].to_numpy())
    f = df["q_id"].map(first).fillna(-1).astype(np.int64).to_numpy()
    s = df["q_id"].map(second).fillna(-1).astype(np.int64).to_numpy()
    return np.where(f == df["d_id"].to_numpy(), s, f)


def _similarity(d_ids: np.ndarray, a_ids: np.ndarray, docs: pd.DataFrame,
                chunk: int = 1_000_000) -> dict[str, np.ndarray]:
    out = {k: np.full(len(d_ids), np.nan, np.float32) for k in ("name_sim", "addr_sim", "num_jac")}
    ok_idx = np.flatnonzero(a_ids >= 0)
    for i in range(0, len(ok_idx), chunk):
        idx = ok_idx[i:i + chunk]
        d = docs.loc[d_ids[idx]]
        a = docs.loc[a_ids[idx]]
        out["name_sim"][idx] = process.cpdist(d["name_core"].tolist(), a["name_core"].tolist(),
                                              scorer=fuzz.token_set_ratio, workers=-1)
        out["addr_sim"][idx] = process.cpdist(d["addr_norm"].tolist(), a["addr_norm"].tolist(),
                                              scorer=fuzz.token_set_ratio, workers=-1)
        jac = []
        for x, y in zip(d["addr_nums"], a["addr_nums"]):
            sx, sy = set(x.split()), set(y.split())
            jac.append(len(sx & sy) / len(sx | sy) if sx and sy else np.nan)
        out["num_jac"][idx] = np.array(jac, np.float32)
    return out


def context_features(kept: pd.DataFrame, p1: np.ndarray, docs: pd.DataFrame) -> pd.DataFrame:
    """kept: q_id, d_id, source. docs: normalized S2+S3 records indexed by d key."""
    df = kept[["q_id", "d_id", "source"]].reset_index(drop=True).assign(p1=p1)
    f = pd.DataFrame({"p1": p1.astype(np.float32)})
    hi = (df["p1"] > 0.5).astype(np.int16)
    gq = df.groupby("q_id")["p1"]
    f["q_p1_max"] = gq.transform("max")
    f["q_p1_sum"] = gq.transform("sum")
    f["q_n_hi"] = hi.groupby(df["q_id"]).transform("sum")
    f["q_p1_rank"] = gq.rank(ascending=False, method="min")
    f["q_gap"] = f["q_p1_max"] - df["p1"]
    gqs = df.groupby(["q_id", "source"])["p1"]
    f["qs_p1_max"] = gqs.transform("max")
    f["qs_n_hi"] = hi.groupby([df["q_id"], df["source"]]).transform("sum")
    per_source = df.groupby(["q_id", "source"])["p1"].max().unstack()
    other = np.where(df["source"].to_numpy() == 2, df["q_id"].map(per_source.get(3, pd.Series(dtype=float))),
                     df["q_id"].map(per_source.get(2, pd.Series(dtype=float))))
    f["other_source_p1_max"] = other
    gd = df.groupby("d_id")["p1"]
    f["d_p1_max"] = gd.transform("max")
    f["d_gap"] = f["d_p1_max"] - df["p1"]
    f["d_n_q"] = gd.transform("size")
    d_ids = df["d_id"].to_numpy()
    for tag, mask in (("any", None),
                      ("same", None),
                      ("other", None)):
        if tag == "any":
            anchor = _anchor(df)
        else:
            parts = np.full(len(df), -1, np.int64)
            for s in (2, 3):
                target = s if tag == "same" else 5 - s
                a = _anchor(df, (df["source"] == target).to_numpy())
                rows = (df["source"] == s).to_numpy()
                parts[rows] = a[rows]
            anchor = parts
        sims = _similarity(d_ids, anchor, docs)
        ap = pd.Series(df["p1"].to_numpy(), index=df["d_id"]).groupby(level=0).max()
        f[f"{tag}_anchor_p1"] = pd.Series(anchor).map(ap).to_numpy(np.float32)
        for k, v in sims.items():
            f[f"{tag}_anchor_{k}"] = v
    return f.astype(np.float32)
