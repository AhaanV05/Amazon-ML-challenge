"""Generate retrieval candidates per country and source. Run: python -m src.candidates SPLIT [FRACTION]"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SEED, WORK_DIR
from .prepare import load_norm
from .retrieve import ROUTE_NAMES, build_indexes, search_all

K = 15
ID_BASE = 1_000_000_000  # numeric id parts are unique per source, < 1e9, no leading zeros (checked)


def q_key(entity_ids: pd.Series) -> np.ndarray:
    return entity_ids.str[3:].astype(np.int64).to_numpy()


def d_key(entity_ids: pd.Series) -> np.ndarray:
    """S2-123 -> 2_000_000_123, S3-123 -> 3_000_000_123."""
    return entity_ids.str[1].astype(np.int64).to_numpy() * ID_BASE + entity_ids.str[3:].astype(np.int64).to_numpy()


def s1_str(keys) -> list[str]:
    return [f"S1-{k}" for k in keys]


def d_str(keys) -> list[str]:
    return [f"S{k // ID_BASE}-{k % ID_BASE}" for k in keys]


def cand_dir(split: str, tag: str) -> Path:
    return WORK_DIR / "cand" / f"{split}_{tag}"


def work_set(fraction: float) -> set[str]:
    """A reproducible random subset of train S1 entities, used for model development."""
    s1 = load_norm("train", 1)[["entity_id"]]
    return set(s1.sample(frac=fraction, random_state=SEED)["entity_id"])


COLUMNS = ["q_id", "d_id"] + [f"{r}_{s}" for r in ROUTE_NAMES for s in ("score", "rank")]
QUERY_CHUNK = 100_000


def generate(split: str, tag: str, subset: set[str] | None = None, k: int = K) -> None:
    out = cand_dir(split, tag)
    out.mkdir(parents=True, exist_ok=True)
    s1 = load_norm(split, 1)
    if subset is not None:
        s1 = s1[s1["entity_id"].isin(subset)]
    schema = pa.schema([("q_id", pa.int64()), ("d_id", pa.int64())]
                       + [(c, pa.float32()) for c in COLUMNS[2:]])
    for source in (2, 3):
        docs = load_norm(split, source)
        for country in sorted(s1["country"].unique()):
            path = out / f"{country}_s{source}.parquet"
            if path.exists():
                continue
            t0 = time.time()
            q = s1[s1["country"] == country].reset_index(drop=True)
            d = docs[docs["country"] == country].reset_index(drop=True)
            tmp = path.with_suffix(".tmp")
            n_pairs = 0
            with pq.ParquetWriter(tmp, schema) as writer:
                if len(d):
                    indexes = build_indexes(d)
                    q_keys, d_keys = q_key(q["entity_id"]), d_key(d["entity_id"])
                    for start in range(0, len(q), QUERY_CHUNK):
                        chunk = q.iloc[start:start + QUERY_CHUNK].reset_index(drop=True)
                        cand = search_all(indexes, chunk, k)
                        cand.insert(0, "d_id", d_keys[cand["d"].to_numpy()])
                        cand.insert(0, "q_id", q_keys[start + cand["q"].to_numpy()])
                        cand = cand.reindex(columns=COLUMNS)
                        writer.write_table(pa.Table.from_pandas(cand, schema=schema, preserve_index=False))
                        n_pairs += len(cand)
                    del indexes
            tmp.replace(path)
            print(f"{split}/{tag} {country} S{source}: {len(q):,} queries x {len(d):,} docs -> {n_pairs:,} pairs "
                  f"({n_pairs / max(len(q), 1):.1f}/query) in {time.time() - t0:.0f}s", flush=True)


def load_candidates(split: str, tag: str) -> pd.DataFrame:
    parts = []
    for path in sorted(cand_dir(split, tag).glob("*.parquet")):
        df = pd.read_parquet(path)
        country, source = path.stem.rsplit("_s", 1)
        parts.append(df.assign(country=country, source=np.int8(source)))
    out = pd.concat(parts, ignore_index=True)
    out["country"] = out["country"].astype("category")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    if split == "train-ce":
        # Entities for training the cross-encoder, disjoint from the work set so its scores there are out-of-sample.
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        work = set(pd.read_parquet(WORK_DIR / "cand" / "work_ids_work20.parquet")["entity_id"])
        pool = load_norm("train", 1)[["entity_id"]]
        pool = pool[~pool["entity_id"].isin(work)]
        ids = set(pool.sample(n=int(frac * (len(pool) + len(work))), random_state=SEED + 1)["entity_id"])
        tag = f"ce{int(frac * 100)}"
        pd.DataFrame({"entity_id": sorted(ids)}).to_parquet(WORK_DIR / "cand" / f"work_ids_{tag}.parquet", index=False)
        generate("train", tag, ids)
    elif split == "train":
        frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        tag = f"work{int(frac * 100)}"
        ids = work_set(frac)
        (WORK_DIR / "cand").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"entity_id": sorted(ids)}).to_parquet(WORK_DIR / "cand" / f"work_ids_{tag}.parquet", index=False)
        generate("train", tag, ids)
    else:
        generate("test", "full")
