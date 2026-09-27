"""v4: retrain both cross-encoders on v3's own candidate pairs, in two folds, then rerun stages 1-2 and predict.

v3 trains LightGBM on 30% of train entities; the rest only supply competition. Their labeled kept pairs (~20M, drawn
from exactly the candidate distribution the models score) are the best cross-encoder training data we have, about
10x what the v3 models saw. To keep every cross-encoder score out-of-sample, non-model entities are split in two
halves: model 0 learns from half 0 and scores half 1, the LightGBM entities and test; model 1 learns from half 1 and
scores half 0. The large models train mostly on pairs the base models are unsure about, the band they rescore.

Run after pipeline_v3 (reuses work/v3, overwrites its cross-encoder scores, models and output files):
    python -m src.pipeline_v4 ce_data ce_train ce_score ce_data_large ce_train ce_large stage1 stage2 predict
    (then optionally: llm_data llm_train llm_score stage1 stage2 predict)
Environment: BER_V4_BASE_PAIRS / BER_V4_LARGE_PAIRS (training pairs per fold), BER_V4_BASE_NAME / BER_V4_LARGE_NAME.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import ID_BASE
from .config import SEED, WORK_DIR
from .pipeline_v3 import (CE_PYTHON, LARGE_BAND, SMOKE, V3, log, roles, step_predict, step_stage1,
                          step_stage2)
from .train import PAIR_BASE

V4 = WORK_DIR / ("v4_smoke" if SMOKE else "v4")
CE = V3 / "ce"
BASE_PAIRS = int(os.environ.get("BER_V4_BASE_PAIRS", 3_000_000))
LARGE_PAIRS = int(os.environ.get("BER_V4_LARGE_PAIRS", 1_200_000))
BASE_NAME = os.environ.get("BER_V4_BASE_NAME", "xlm-roberta-base")
LARGE_NAME = os.environ.get("BER_V4_LARGE_NAME", "BAAI/bge-reranker-v2-m3")
SCRIPT = str(Path(__file__).with_name("cross_encoder.py"))
WORKERS = os.environ.get("BER_V4_WORKERS", "16")  # tokenizer workers per cross-encoder job
BATCH_DIV = int(os.environ.get("BER_V4_BATCH_DIV", 1))  # >1 only for dry runs on a small GPU
# v5 stage: a 4B LLM reranker (LoRA) on pairs both cross-encoders are unsure about.
LLM_NAME = os.environ.get("BER_V4_LLM_NAME", "Qwen/Qwen3-Reranker-4B")
LLM_SCRIPT = str(Path(__file__).with_name("llm_reranker.py"))
LLM_PAIRS = int(os.environ.get("BER_V4_LLM_PAIRS", 60_000))  # training pairs per fold
LLM_MAX_PAIRS = int(os.environ.get("BER_V4_LLM_MAX_PAIRS", 700_000))  # pairs scored, train and test together
LLM_LOGIT = float(np.log(19))  # large-model probability between 0.05 and 0.95
GPUS = int(os.environ.get("BER_V4_GPUS", 1))  # >1: fold models train side by side and scoring is split across GPUs


def _half(q_ids: np.ndarray) -> np.ndarray:
    """0/1 for cross-encoder training entities (non-model), -1 for LightGBM entities."""
    r = roles()
    half = pd.Series((pd.util.hash_array(r["q_id"].to_numpy(np.int64) * 2 + 1) % 2).astype(np.int8),
                     index=r["q_id"].to_numpy())
    half[r["is_model"].to_numpy()] = -1
    return half.reindex(q_ids).fillna(-1).to_numpy(np.int8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x.astype(np.float64)))


def _link_records(split: str, folder: Path) -> None:
    dst = folder / f"records_{split}.parquet"
    if not dst.exists():
        try:
            os.symlink(CE / f"records_{split}.parquet", dst)
        except OSError:  # Windows without symlink rights (laptop smoke test)
            shutil.copy(CE / f"records_{split}.parquet", dst)


def _on_gpus(commands: list[list[str]]) -> None:
    """Run the commands GPUS at a time, each process seeing one GPU; raise if any fails."""
    free = list(range(GPUS))
    running = []
    for cmd in commands:
        if not free:
            gpu, p = running.pop(0)
            if p.wait():
                raise subprocess.CalledProcessError(p.returncode, p.args)
            free.append(gpu)
        gpu = free.pop(0)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if GPUS > 1 else None
        running.append((gpu, subprocess.Popen(cmd, env=env)))
    for _, p in running:
        if p.wait():
            raise subprocess.CalledProcessError(p.returncode, p.args)


def _score_cmd(model: Path, folder: Path, pairs: str, records: str, batch: int, script: str = SCRIPT) -> list[str]:
    return [CE_PYTHON, script, "score", "--data", str(folder), "--out", str(model), "--pairs", pairs,
            "--records", records, "--workers", WORKERS, "--infer-batch", str(max(1, batch // BATCH_DIV))]


# ---------------------------------------------------------------------------------------------------------------
def _write_fold(kept: pd.DataFrame, idx: np.ndarray, name: str, t0: float) -> None:
    folder = V4 / name
    folder.mkdir(parents=True, exist_ok=True)
    k = kept.iloc[np.sort(idx)]
    pd.DataFrame({"q_rec": k["q_id"].to_numpy(np.int64) + ID_BASE, "d_rec": k["d_id"].to_numpy(np.int64),
                  "label": k["label"].to_numpy(np.int8)}).to_parquet(folder / "ce_pairs.parquet", index=False)
    _link_records("train", folder)
    log(f"ce data {name}: {len(idx):,} pairs, positive {k['label'].mean():.1%} | {time.time() - t0:.0f}s")


def step_ce_data() -> None:
    """Training pairs for the two base models: a random sample of each half's kept pairs."""
    t0 = time.time()
    kept = pd.read_parquet(V3 / "train_kept.parquet", columns=["q_id", "d_id", "label"])
    half = _half(kept["q_id"].to_numpy())
    rng = np.random.default_rng(SEED)
    for f in (0, 1):
        pool = np.flatnonzero(half == f)
        _write_fold(kept, rng.choice(pool, size=min(BASE_PAIRS, len(pool)), replace=False), f"fold{f}", t0)


def step_ce_data_large() -> None:
    """Training pairs for the two large models: the pairs the (out-of-sample) base scores leave uncertain, the band
    the large models rescore, topped up with random pairs. Runs after ce_score."""
    t0 = time.time()
    kept = pd.read_parquet(V3 / "train_kept.parquet", columns=["q_id", "d_id", "label"])
    s = pd.read_parquet(CE / "train_scores.parquet")
    logit = pd.Series(s["ce_logit"].to_numpy(), index=(s["q_rec"].to_numpy() - ID_BASE) * PAIR_BASE
                      + s["d_rec"].to_numpy())
    p = _sigmoid(logit.reindex(kept["q_id"].to_numpy(np.int64) * PAIR_BASE + kept["d_id"].to_numpy()).to_numpy())
    half = _half(kept["q_id"].to_numpy())
    rng = np.random.default_rng(SEED + 1)
    for f in (0, 1):
        pool = np.flatnonzero(half == f)
        band = pool[(p[pool] >= LARGE_BAND[0]) & (p[pool] <= LARGE_BAND[1])]
        band = rng.choice(band, size=min(LARGE_PAIRS, len(band)), replace=False)
        rest = np.setdiff1d(pool, band)
        top_up = rng.choice(rest, size=min(max(LARGE_PAIRS - len(band), 0), len(rest)), replace=False)
        _write_fold(kept, np.concatenate([band, top_up]), f"fold{f}_large", t0)
        log(f"  fold {f}: {len(pool):,} pairs available, {len(band):,} from the uncertain band")


def step_ce_train() -> None:
    """Trains every model whose training data exists and that is not trained yet (base first, large after
    ce_data_large)."""
    jobs = [(f"fold{f}", BASE_NAME, f"base{f}", str(256 // BATCH_DIV), "3e-5", BASE_PAIRS) for f in (0, 1)] + \
           [(f"fold{f}_large", LARGE_NAME, f"large{f}", str(128 // BATCH_DIV), "2e-5", LARGE_PAIRS) for f in (0, 1)]
    for data, model, out, batch, lr, n in jobs:
        if (V4 / out / "metrics.json").exists() or not (V4 / data / "ce_pairs.parquet").exists():
            continue
        t0 = time.time()
        subprocess.run([CE_PYTHON, SCRIPT, "train", "--data", str(V4 / data), "--model", model, "--out", str(V4 / out),
                        "--batch", batch, "--lr", lr, "--max-train-pairs", str(n), "--workers", WORKERS], check=True)
        log(f"trained {out} ({model}) in {(time.time() - t0) / 60:.0f} min: {(V4 / out / 'metrics.json').read_text()}")


def _route(split: str, pairs: pd.DataFrame) -> np.ndarray:
    """Which fold model scores each pair: model 1 for half-0 train entities, model 0 for everything else."""
    if split == "test":
        return np.zeros(len(pairs), dtype=np.int8)
    return (_half(pairs["q_rec"].to_numpy(np.int64) - ID_BASE) == 0).astype(np.int8)


def _score_routed(split: str, pairs: pd.DataFrame, kind: str, batch: int) -> pd.DataFrame:
    """Score pairs with the fold models chosen by _route; kind is "base", "large" or "llm"."""
    route = _route(split, pairs)
    tag = {"base": "", "large": "_large", "llm": "_llm"}[kind]
    names, commands = [], []
    for m in (0, 1):
        sub = pairs.loc[route == m, ["q_rec", "d_rec"]]
        if sub.empty:
            continue
        # one chunk per GPU share, so every GPU gets a similar number of pairs
        chunks = max(1, round(GPUS * len(sub) / len(pairs)))
        for c, idx in enumerate(np.array_split(np.arange(len(sub)), chunks)):
            name = f"{split}{tag}_m{m}_c{c}_pairs.parquet"
            sub.iloc[idx].to_parquet(CE / name, index=False)
            names.append(name)
            commands.append(_score_cmd(V4 / f"{kind}{m}", CE, name, f"records_{split}.parquet", batch,
                                       LLM_SCRIPT if kind == "llm" else SCRIPT))
        log(f"{split} {kind} model {m}: {len(sub):,} pairs in {chunks} chunk(s)")
    _on_gpus(commands)
    return pd.concat([pd.read_parquet(CE / n.replace("_pairs", "_scores")) for n in names], ignore_index=True)


def step_ce_score() -> None:
    t0 = time.time()
    for split in ("train", "test"):
        pairs = pd.read_parquet(CE / f"{split}_pairs.parquet")
        _score_routed(split, pairs, "base", 4096).to_parquet(CE / f"{split}_scores.parquet", index=False)
        log(f"ce score {split}: {len(pairs):,} pairs | {time.time() - t0:.0f}s")
    log("CE DONE")


def step_ce_large() -> None:
    t0 = time.time()
    for split in ("train", "test"):
        s = pd.read_parquet(CE / f"{split}_scores.parquet")
        p = _sigmoid(s["ce_logit"].to_numpy())
        band = s.loc[(p >= LARGE_BAND[0]) & (p <= LARGE_BAND[1]), ["q_rec", "d_rec"]].reset_index(drop=True)
        log(f"large cross-encoder {split}: {len(band):,} uncertain pairs ({len(band) / len(s):.1%})")
        _score_routed(split, band, "large", 1024).to_parquet(CE / f"{split}_large_scores.parquet", index=False)
        log(f"ce large {split} | {time.time() - t0:.0f}s")
    log("CE LARGE DONE")


def _llm_band() -> dict[str, pd.DataFrame]:
    """Pairs both cross-encoders are unsure about (large-model probability 0.05-0.95), capped at LLM_MAX_PAIRS by one
    shared |logit| threshold, so train and test follow the same rule."""
    within = {}
    for split in ("train", "test"):
        s = pd.read_parquet(CE / f"{split}_large_scores.parquet")
        within[split] = s[np.abs(s["ce_logit"].to_numpy()) <= LLM_LOGIT].reset_index(drop=True)
    total = sum(len(f) for f in within.values())
    if total > LLM_MAX_PAIRS:
        t = np.quantile(np.abs(np.concatenate([f["ce_logit"].to_numpy() for f in within.values()])), LLM_MAX_PAIRS / total)
        within = {k: f[np.abs(f["ce_logit"].to_numpy()) <= t].reset_index(drop=True) for k, f in within.items()}
    return within


def step_llm_data() -> None:
    kept = pd.read_parquet(V3 / "train_kept.parquet", columns=["q_id", "d_id", "label"])
    label = pd.Series(kept["label"].to_numpy(np.int8), index=kept["q_id"].to_numpy(np.int64) * PAIR_BASE
                      + kept["d_id"].to_numpy(np.int64))
    band = _llm_band()["train"]
    wide = pd.read_parquet(CE / "train_large_scores.parquet")  # top-up pool: everything the large model saw
    rng = np.random.default_rng(SEED)
    for f in (0, 1):
        pool = band[_half(band["q_rec"].to_numpy(np.int64) - ID_BASE) == f]
        pick = pool.iloc[rng.choice(len(pool), size=min(LLM_PAIRS, len(pool)), replace=False)]
        if len(pick) < LLM_PAIRS:
            extra = wide[_half(wide["q_rec"].to_numpy(np.int64) - ID_BASE) == f]
            extra = extra[~extra.set_index(["q_rec", "d_rec"]).index.isin(pick.set_index(["q_rec", "d_rec"]).index)]
            pick = pd.concat([pick, extra.iloc[rng.choice(len(extra), size=min(LLM_PAIRS - len(pick), len(extra)),
                                                          replace=False)]])
        keys = (pick["q_rec"].to_numpy(np.int64) - ID_BASE) * PAIR_BASE + pick["d_rec"].to_numpy(np.int64)
        folder = V4 / f"fold{f}_llm"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"q_rec": pick["q_rec"].to_numpy(np.int64), "d_rec": pick["d_rec"].to_numpy(np.int64),
                      "label": label.reindex(keys).to_numpy()}).to_parquet(folder / "ce_pairs.parquet", index=False)
        _link_records("train", folder)
        log(f"llm data fold {f}: {len(pick):,} pairs ({len(pool):,} in the band for this half)")


def step_llm_train() -> None:
    t0 = time.time()
    todo = [f for f in (0, 1) if not (V4 / f"llm{f}" / "metrics.json").exists()]
    _on_gpus([[CE_PYTHON, LLM_SCRIPT, "train", "--data", str(V4 / f"fold{f}_llm"), "--model", LLM_NAME,
               "--out", str(V4 / f"llm{f}"), "--max-train-pairs", str(LLM_PAIRS), "--batch", str(max(1, 32 // BATCH_DIV))]
              for f in todo])
    for f in (0, 1):
        log(f"llm{f} ({LLM_NAME}) {'trained' if f in todo else 'already trained'} | {(time.time() - t0) / 60:.0f} min: "
            f"{(V4 / f'llm{f}' / 'metrics.json').read_text()}")


def step_llm_score() -> None:
    t0 = time.time()
    for split, band in _llm_band().items():
        log(f"llm reranker {split}: {len(band):,} pairs")
        _score_routed(split, band, "llm", 128).to_parquet(CE / f"{split}_llm_scores.parquet", index=False)
        log(f"llm score {split} | {time.time() - t0:.0f}s")
    log("LLM DONE")


STEPS = {"ce_data": step_ce_data, "ce_train": step_ce_train, "ce_score": step_ce_score,
         "ce_data_large": step_ce_data_large, "ce_large": step_ce_large,
         "llm_data": step_llm_data, "llm_train": step_llm_train, "llm_score": step_llm_score,
         "stage1": step_stage1, "stage2": step_stage2, "predict": step_predict}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for step in sys.argv[1:]:
        log(f"=== v4 {step} ===")
        STEPS[step]()
