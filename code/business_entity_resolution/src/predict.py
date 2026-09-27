"""Test inference.

python -m src.candidates test   # retrieval candidates
python -m src.predict filter    # learned filter -> work/stage/test_kept.parquet (= candidate_pairs.tsv)
[optional] python -m src.ce_data test, then score test_pairs on the GPU with src/cross_encoder.py
python -m src.predict score     # matcher (stage 1 + 2), decoding, both TSVs, official validator
"""
import json
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import d_str
from .ce_data import CE_DIR, attach_ce
from .config import DATA_DIR, OUTPUT_DIR, WORK_DIR
from .context import context_features
from .data import load_table
from .decode import decode_expected_f, decode_threshold, one_owner
from .filtering import filter_candidates
from .lgbutil import predict_chunked, predict_frames
from .submission import write_submission
from .train import MODEL_DIR, matcher_features

KEPT = WORK_DIR / "stage" / "test_kept.parquet"


def step_filter() -> None:
    kept = filter_candidates("test", "full")
    KEPT.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(KEPT, index=False)
    per_q = kept.groupby("q_id").size()
    print(f"test kept pairs {len(kept):,} | mean {len(kept) / load_table('test', 'source1').num_rows:.2f} per S1 entity, "
          f"p95 {per_q.quantile(0.95):.0f}", flush=True)


def step_score() -> None:
    t0 = time.time()
    kept = pd.read_parquet(KEPT)
    stage1_path, stage2_path = MODEL_DIR / "stage1.txt", MODEL_DIR / "stage2.txt"
    use_stage2 = stage2_path.exists()
    if use_stage2:
        stage1 = lgb.Booster(model_file=str(stage1_path))
        stage2 = lgb.Booster(model_file=str(stage2_path))
        use_ce = "ce_logit" in stage1.feature_name()
        if use_ce and not (CE_DIR / "test_scores.parquet").exists():
            raise SystemExit("models use the cross-encoder but work/ce/test_scores.parquet is missing")
        chosen = json.loads((MODEL_DIR / "validation2.json").read_text())["results"]["stage2"]
    else:
        matcher = lgb.Booster(model_file=str(MODEL_DIR / "matcher.txt"))
        chosen = json.loads((MODEL_DIR / "validation.json").read_text())["results"]["one_owner"]
    p = np.empty(len(kept))
    # Countries share no entities or records, so scoring them separately is exact and bounds memory.
    for country, idx in kept.groupby("country", observed=True).indices.items():
        kc = kept.iloc[idx]
        mf = matcher_features(kc, "test")
        if use_stage2:
            if use_ce:
                mf["ce_logit"] = attach_ce(kc, "test")
            p1 = predict_chunked(stage1, mf)
            from .train2 import docs_frame
            ctx = context_features(kc, p1, docs_frame("test", kc["d_id"].unique()))
            p[idx] = predict_frames(stage2, [mf, ctx])
        else:
            p[idx] = predict_chunked(matcher, mf)
        print(f"  scored {country}: {len(kc):,} pairs | {time.time() - t0:.0f}s", flush=True)
        del mf
    kept["p"] = p
    print(f"scored {len(kept):,} pairs (stage 2: {use_stage2}) | {time.time() - t0:.0f}s", flush=True)

    resolved = one_owner(kept)
    if chosen["expected_f"]["macro_f05"] > chosen["threshold_f05"]:
        matches, how = decode_expected_f(resolved), "expected-F0.5"
    else:
        matches, how = decode_threshold(resolved, chosen["best_threshold"]), f"threshold {chosen['best_threshold']}"
    s1_ids = load_table("test", "source1")["entity_id"].to_pylist()
    cands = {f"S1-{q}": d_str(ds) for q, ds in kept.groupby("q_id")["d_id"].agg(list).items()}
    preds = {f"S1-{q}": d_str(ds) for q, ds in matches.items()}
    write_submission(OUTPUT_DIR, s1_ids, preds, cands)
    n_c = sum(len(v) for v in cands.values())
    n_m = sum(len(v) for v in preds.values())
    print(f"decoder {how} | mean candidates per S1 {n_c / len(s1_ids):.2f} | mean matches per S1 {n_m / len(s1_ids):.2f} "
          f"| empty predictions {1 - len(preds) / len(s1_ids):.1%} | {time.time() - t0:.0f}s", flush=True)
    for country, g in kept.groupby("country", observed=True):
        mq = g[g["q_id"].isin([int(k[3:]) for k in preds])]["q_id"].nunique() if len(preds) else 0
        print(f"  {country}: {g['q_id'].nunique():,} entities with candidates, {mq:,} with matches", flush=True)
    validator = DATA_DIR.parent / "utils" / "validate_submission.py"
    subprocess.run([sys.executable, str(validator), "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
                    "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(DATA_DIR / "test")],
                   check=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "score"
    step_filter() if step == "filter" else step_score()
