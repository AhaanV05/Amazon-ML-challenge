"""Export data for the cross-encoder (run on the laptop), and join its scores back.

python -m src.ce_data train   # ce_pairs (labelled, cross-encoder training set), work_pairs, records_train
python -m src.ce_data test    # test_pairs, records_test (needs work/stage/test_kept.parquet)

Copy work/ce/ to Google Drive, run src/cross_encoder.py on Colab, copy *_scores.parquet back to work/ce/.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import ID_BASE
from .config import WORK_DIR
from .data import load_table

CE_DIR = WORK_DIR / "ce"
STAGE_DIR = WORK_DIR / "stage"
PAIR_BASE = 4 * ID_BASE


def record_texts(split: str, keys: np.ndarray) -> pd.DataFrame:
    """key = source * 1e9 + numeric id; text = lowercased 'name | address' in the original script."""
    wanted = np.unique(keys)
    parts = []
    for s in (1, 2, 3):
        t = load_table(split, f"source{s}").to_pandas()
        k = s * ID_BASE + t["entity_id"].str[3:].astype(np.int64).to_numpy()
        m = np.isin(k, wanted)
        sub = t[m]
        text = (sub["business_name"] + " | " + sub["business_address"]).str.lower().str.split().str.join(" ")
        parts.append(pd.DataFrame({"key": k[m], "text": text.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def to_pairs(frame: pd.DataFrame, with_label: bool = False) -> pd.DataFrame:
    out = pd.DataFrame({"q_rec": frame["q_id"].to_numpy(np.int64) + ID_BASE, "d_rec": frame["d_id"].to_numpy(np.int64)})
    if with_label:
        out["label"] = frame["label"].to_numpy(np.int8)
    return out


def attach_ce(frame: pd.DataFrame, name: str) -> np.ndarray:
    """ce_logit for each (q_id, d_id) row of frame from work/ce/{name}_scores.parquet (NaN if unscored)."""
    s = pd.read_parquet(CE_DIR / f"{name}_scores.parquet")
    keys = (s["q_rec"].to_numpy(np.int64) - ID_BASE) * PAIR_BASE + s["d_rec"].to_numpy(np.int64)
    lookup = pd.Series(s["ce_logit"].to_numpy(np.float32), index=keys)
    want = frame["q_id"].to_numpy(np.int64) * PAIR_BASE + frame["d_id"].to_numpy(np.int64)
    return lookup.reindex(want).to_numpy(np.float32)


def _copy_script() -> None:
    # The Colab notebook runs this exact file, so the repo stays the single source of truth.
    shutil.copy(Path(__file__).with_name("cross_encoder.py"), CE_DIR / "cross_encoder.py")


def export_train() -> None:
    from .filtering import filter_candidates
    from .train import add_labels, truth_sets
    t0 = time.time()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    _copy_script()
    kept = filter_candidates("train", "ce10")
    ids = pd.read_parquet(WORK_DIR / "cand" / "work_ids_ce10.parquet")["entity_id"]
    truth = truth_sets(set(ids.str[3:].astype(np.int64).tolist()))
    kept = add_labels(kept, truth)
    ce = to_pairs(kept, with_label=True)
    ce.to_parquet(CE_DIR / "ce_pairs.parquet", index=False)
    work = to_pairs(pd.read_parquet(STAGE_DIR / "work20.parquet", columns=["q_id", "d_id"]))
    work.to_parquet(CE_DIR / "work_pairs.parquet", index=False)
    keys = np.concatenate([ce["q_rec"], ce["d_rec"], work["q_rec"], work["d_rec"]])
    record_texts("train", keys).to_parquet(CE_DIR / "records_train.parquet", index=False)
    print(f"cross-encoder training pairs {len(ce):,} ({ce['label'].mean():.1%} positive) | work pairs {len(work):,} "
          f"| {time.time() - t0:.0f}s", flush=True)


def export_test() -> None:
    t0 = time.time()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    _copy_script()
    test = to_pairs(pd.read_parquet(STAGE_DIR / "test_kept.parquet", columns=["q_id", "d_id"]))
    test.to_parquet(CE_DIR / "test_pairs.parquet", index=False)
    record_texts("test", np.concatenate([test["q_rec"], test["d_rec"]])).to_parquet(CE_DIR / "records_test.parquet",
                                                                                   index=False)
    print(f"test pairs {len(test):,} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    export_train() if (sys.argv[1] if len(sys.argv) > 1 else "train") == "train" else export_test()
