"""Build colab/pipeline_v3.ipynb with the current source code embedded, so Colab needs no code upload.

Run after any code change:  python colab/build_notebook.py
"""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "pipeline_v3.ipynb"


def code_blob() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted([*ROOT.glob("src/*.py"), ROOT / "requirements.txt"]):
            z.write(p, p.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


# Shared by the unattended cells: run a pipeline step with logging, copy files to Drive.
HELPERS = """import os, sys, time, subprocess, glob, shutil
os.environ['BER_CE_PYTHON'] = sys.executable
# Colab ships torchao 0.10, which current peft refuses to import alongside; nothing here needs torchao.
!pip uninstall -y -q torchao
!pip install -q peft
!{sys.executable} -c "import peft; print('peft', peft.__version__)"
LOG = '/content/work/run.log'

def run(module, steps):
    ok = subprocess.run(f'cd /content/code && {PY} -m src.{module} {steps} >> {LOG} 2>&1', shell=True).returncode == 0
    print(time.strftime('%H:%M:%S'), 'OK' if ok else 'FAILED', module, steps, flush=True)
    shutil.copy(LOG, DRIVE)
    return ok

def to_drive(patterns, dest):
    try:
        for pattern in patterns:
            for src in glob.glob(pattern):
                if os.path.isfile(src):
                    dst = os.path.join(DRIVE, dest, os.path.relpath(src, '/content/work'))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy(src, dst)
        print(time.strftime('%H:%M:%S'), 'saved to Drive:', dest, flush=True)
    except OSError as e:
        print('Drive copy failed, continuing:', e, flush=True)

def outputs(name):
    to_drive(['/content/work/output/*', '/content/work/v3/validation.json', '/content/work/v3/val_pairs.parquet',
              '/content/work/v3/filter_*'], name)
"""

# One unattended cell for a fresh runtime: search + filter + features, then v4, then v5.
RUN_ALL = """# EVERYTHING, unattended (~7 h). Search, filter and features as in v3; v4: both cross-encoders retrained on our own
# candidate pairs in two folds (every score stays out-of-sample); v5: a 4B LLM reranker (Qwen3-Reranker-4B + LoRA) on
# the pairs both cross-encoders are unsure about. Each version is saved to Drive as soon as it exists; checkpoints and
# the log (run.log) go to Drive after every step. The runtime is released at the end.
""" + HELPERS + """
ok = run('pipeline_v3', 'prepare candidates filter ce_export')
if ok:
    to_drive(['/content/work/v3/*.parquet', '/content/work/v3/*.json', '/content/work/v3/*.csv', '/content/work/v3/*.txt',
              '/content/work/v3/ce/*.parquet'], 'ckpt')
    features = subprocess.Popen(f'cd /content/code && {PY} -m src.pipeline_v3 features >> /content/work/features.log 2>&1',
                                shell=True)  # CPU, alongside the GPU training
    ok = run('pipeline_v4', 'ce_data ce_train ce_score')
    ok = features.wait() == 0 and ok
    shutil.copy('/content/work/features.log', DRIVE)
if ok:
    to_drive(['/content/work/v3/ce/*_scores.parquet'], 'ckpt')
    ok = run('pipeline_v4', 'ce_data_large ce_train ce_large stage1 stage2 predict')
if ok:
    to_drive(['/content/work/v3/ce/*_scores.parquet'], 'ckpt')
    outputs('out_v4')
    if run('pipeline_v4', 'llm_data llm_train llm_score stage1 stage2 predict'):
        outputs('out_v5')
print(time.strftime('%H:%M:%S'), 'finished; releasing the runtime', flush=True)
from google.colab import runtime
runtime.unassign()"""

# v5 alone on a fresh runtime, resumed from the Drive checkpoint the full run left (kept pairs, cross-encoder scores).
RUN_V5 = """# v5 ONLY, unattended (~3-4 h): restore the checkpoint, normalize again (1 min) and recompute pair features on the
# CPU while the GPU trains and runs the LLM reranker; then stages 1-2 and predict. LoRA adapters and LLM scores go to
# Drive as soon as they exist (a rerun after a disconnect skips the trained folds). The runtime is released at the end.
""" + HELPERS + """
!mkdir -p /content/work/v3 /content/work/v4 && cp -r {DRIVE}/ckpt/v3/. /content/work/v3/ && ls /content/work/v3 /content/work/v3/ce
!cp -r {DRIVE}/ckpt/v4/. /content/work/v4/ 2>/dev/null; ls /content/work/v4
ok = run('pipeline_v3', 'prepare')
if ok:
    features = subprocess.Popen(f'cd /content/code && {PY} -m src.pipeline_v3 features >> /content/work/features.log 2>&1',
                                shell=True)  # CPU, alongside the GPU
    ok = run('pipeline_v4', 'llm_data llm_train')
    to_drive(['/content/work/v4/llm*/*'], 'ckpt')
    ok = ok and run('pipeline_v4', 'llm_score')
    to_drive(['/content/work/v3/ce/*_llm_scores.parquet'], 'ckpt')
    ok = features.wait() == 0 and ok
    shutil.copy('/content/work/features.log', DRIVE)
if ok and run('pipeline_v4', 'stage1 stage2 predict'):
    outputs('out_v5')
print(time.strftime('%H:%M:%S'), 'finished; releasing the runtime', flush=True)
from google.colab import runtime
runtime.unassign()"""


# v6: the matcher (LightGBM stages 1-2) learns from ALL train entities; resumes from the Drive checkpoint.
RUN_V6 = """# v6, unattended (~1.3 h): restore the checkpoint (kept pairs, cross-encoder scores, and the LLM scores once v5 has
# saved them), normalize, recompute features, then stages 1-2 on 100% of train entities and predict. Results go to
# ber_v3/out_v6 (or out_v6_llm when the LLM scores were there); log in ber_v3/run_v6.log. Releases the runtime.
""" + HELPERS + """
LOG = '/content/work/run_v6.log'
!mkdir -p /content/work/v3 && cp -r {DRIVE}/ckpt/v3/. /content/work/v3/ && ls /content/work/v3/ce
os.environ['BER_STAGE_ALL'] = '1'
name = 'out_v6_llm' if len(glob.glob('/content/work/v3/ce/*_llm_scores.parquet')) == 2 else 'out_v6'
print('results will go to', name, flush=True)
if run('pipeline_v3', 'prepare features') and run('pipeline_v4', 'stage1 stage2 predict'):
    outputs(name)
    to_drive(['/content/work/v3/p1_test.npy'], name)
print(time.strftime('%H:%M:%S'), 'finished; releasing the runtime', flush=True)
from google.colab import runtime
runtime.unassign()"""


CELLS = [
    md("""# Full pipeline v3 on the G4 machine (48 cores, 176 GB RAM, RTX PRO 6000)

Every train S1 entity goes through retrieval, the candidate filter, the cross-encoder and feature computation,
exactly like test, so the competition features match at train and test time.

**Drive layout needed** (no code upload; the code is embedded in cell 2):
```
My Drive/ber_ce/ce_model/                trained cross-encoder
My Drive/ber_v3/parquet/*.parquet        7 raw data files (work/parquet on the laptop)
My Drive/ber_v3/work_ids_ce10.parquet    (work/cand on the laptop)
```
Run the cells top to bottom and **save the notebook after each cell** so progress is visible on the laptop.
Rough timings: setup 5 min, prepare + candidates ~60 min, filter ~25 min, CE scoring ~40 min (background, while
features compute), stage 1 ~25 min, stage 2 ~30 min, predict ~10 min.
"""),
    code("""!nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/ber_v3'
CE_DRIVE = '/content/drive/MyDrive/ber_ce/ce_model'
!ls -la {DRIVE} {DRIVE}/parquet {CE_DRIVE}"""),
    code("""# Unpack the pipeline code embedded in the cell above (rebuilt from the repo by colab/build_notebook.py).
import base64, io, zipfile, shutil, os
shutil.rmtree('/content/code', ignore_errors=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall('/content/code')
print(sorted(os.listdir('/content/code/src')))"""),
    code("""# Setup: data, cross-encoder model, and a separate environment with the pinned packages.
import os, sys
!mkdir -p /content/work/parquet /content/work/cand /content/work/output
!cp -n {DRIVE}/parquet/*.parquet /content/work/parquet/ && cp -n {DRIVE}/work_ids_ce10.parquet /content/work/cand/ && ls -la /content/work/parquet
!rm -rf /content/work/ce_model && cp -r {CE_DRIVE} /content/work/ce_model && ls /content/work/ce_model
!pip install -q uv && uv venv -q --allow-existing /content/venv && uv pip install -q --python /content/venv/bin/python -r /content/code/requirements.txt
os.environ.update({'BER_WORK_DIR': '/content/work', 'BER_OUTPUT_DIR': '/content/work/output', 'BER_THREADS': '46',
                   'BER_CE_MODEL': '/content/work/ce_model', 'BER_CE_PYTHON': sys.executable,
                   'PYTHONIOENCODING': 'utf-8', 'BER_K': '20', 'BER_CAND_BUDGET': '16', 'BER_MODEL_FRAC': '0.3'})
PY = '/content/venv/bin/python'
!{PY} -c "import pandas, lightgbm, sparse_dot_topn, rapidfuzz; print('pipeline env ok', pandas.__version__, lightgbm.__version__)\""""),
    code("""# 0. GPU, in the background: train the larger cross-encoder (bge-reranker-v2-m3, Apache-2.0, 568M) while the CPUs
#    run search. It later rescores only the pairs the base model is unsure about. Optional: skipped if it fails.
!mkdir -p /content/ce_large && cp -n /content/drive/MyDrive/ber_ce/ce_pairs.parquet /content/drive/MyDrive/ber_ce/records_train.parquet /content/ce_large/
# Popen, not `!... &`: in Colab a backgrounded `!` command still blocks the cell until it exits.
import subprocess
subprocess.Popen(f'cd /content/code && exec {sys.executable} src/cross_encoder.py train --data /content/ce_large --model BAAI/bge-reranker-v2-m3 --out /content/work/ce_large_model --batch 128 --lr 2e-5 --max-train-pairs 1200000 --workers 8 > /content/work/ce_large_train.log 2>&1', shell=True, start_new_session=True)
!sleep 240; tail -4 /content/work/ce_large_train.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"""),
    code("""# 1. Normalize every record, then retrieval candidates for ALL train and test S1 entities (top-20 per route).
!cd /content/code && {PY} -m src.pipeline_v3 prepare candidates 2>&1 | tee -a /content/work/v3.log"""),
    code("""# 2. Candidate filter: trained on model entities, features over the whole population; prints the size/ceiling sweep.
!cd /content/code && {PY} -m src.pipeline_v3 filter 2>&1 | tee -a /content/work/v3.log"""),
    code("""# 3. Export texts for the cross-encoder, then score every kept pair on the GPU IN THE BACKGROUND.
!cd /content/code && {PY} -m src.pipeline_v3 ce_export 2>&1 | tee -a /content/work/v3.log
import subprocess
subprocess.Popen(f'cd /content/code && exec {PY} -m src.pipeline_v3 ce_score > /content/work/ce_score.log 2>&1', shell=True, start_new_session=True)
!sleep 90; tail -3 /content/work/ce_score.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"""),
    code("""# 4. Pair features on the CPU while the GPU scores.
!cd /content/code && {PY} -m src.pipeline_v3 features 2>&1 | tee -a /content/work/v3.log"""),
    code("""# 5. Wait for the cross-encoder to finish (prints its log).
!until grep -qE "CE DONE|Traceback|Error" /content/work/ce_score.log; do sleep 30; done; tail -8 /content/work/ce_score.log"""),
    code("""# 5b. Wait for the large cross-encoder to finish training (its files are saved last), then rescore uncertain pairs.
!until [ -f /content/work/ce_large_model/metrics.json ] || grep -qE "Traceback|Error" /content/work/ce_large_train.log; do sleep 30; done; tail -12 /content/work/ce_large_train.log
!cd /content/code && {PY} -m src.pipeline_v3 ce_large 2>&1 | tee -a /content/work/v3.log"""),
    code("""# 6. Stage 1 (5-fold out-of-fold on model entities), stage 2 (context features, validation report), test predictions.
!cd /content/code && {PY} -m src.pipeline_v3 stage1 stage2 predict 2>&1 | tee -a /content/work/v3.log"""),
    code("""# 7. Copy the submission files and reports to Drive.
!mkdir -p {DRIVE}/out && cp /content/work/output/*.tsv /content/work/v3/validation.json /content/work/v3/filter_sweep.csv /content/work/v3/filter_op.json /content/work/v3.log {DRIVE}/out/ && ls -la {DRIVE}/out"""),
    code("""# Optional: back up models and small artifacts to Drive (useful if the runtime disconnects).
!mkdir -p {DRIVE}/v3_backup && cp /content/work/v3/*.txt /content/work/v3/*.json /content/work/v3/*.npy /content/work/v3/roles.parquet {DRIVE}/v3_backup/ 2>/dev/null; ls -la {DRIVE}/v3_backup"""),
]

RUN_CELLS = [
    md("""# Full run on the G4 machine: v3 core, then v4 and v5 (one unattended cell)

Needs on Drive: `My Drive/ber_v3/parquet/*.parquet` (7 raw data files) and `My Drive/ber_v3/work_ids_ce10.parquet`.
Run cells 1-4, then cell 5 (~7 h). Results appear on Drive as they finish: `ber_v3/out_v4/`, then `ber_v3/out_v5/`;
`ber_v3/run.log` is updated after every step.
"""),
    CELLS[1],
    CELLS[2],
    code("""# Setup: data and a separate environment with the pinned packages.
import os, sys
!mkdir -p /content/work/parquet /content/work/cand /content/work/output
!cp -n {DRIVE}/parquet/*.parquet /content/work/parquet/ && cp -n {DRIVE}/work_ids_ce10.parquet /content/work/cand/ && ls -la /content/work/parquet
!pip install -q uv && uv venv -q --allow-existing /content/venv && uv pip install -q --python /content/venv/bin/python -r /content/code/requirements.txt
os.environ.update({'BER_WORK_DIR': '/content/work', 'BER_OUTPUT_DIR': '/content/work/output', 'BER_THREADS': '46',
                   'BER_CE_PYTHON': sys.executable, 'PYTHONIOENCODING': 'utf-8', 'BER_K': '20',
                   'BER_CAND_BUDGET': '16', 'BER_MODEL_FRAC': '0.3'})
PY = '/content/venv/bin/python'
!{PY} -c "import pandas, lightgbm, sparse_dot_topn, rapidfuzz; print('pipeline env ok', pandas.__version__, lightgbm.__version__)\""""),
    code(RUN_ALL),
]

V5_CELLS = [
    md("""# v5 only: resume from the Drive checkpoint and add the LLM reranker

Needs on Drive: `My Drive/ber_v3/parquet/*.parquet`, `My Drive/ber_v3/work_ids_ce10.parquet` and `My Drive/ber_v3/ckpt/`
(left by run_v4.ipynb). Run cells 1-4, then cell 5 (~3-4 h). The result appears on Drive in `ber_v3/out_v5/`;
`ber_v3/run.log` is updated after every step.
"""),
    *RUN_CELLS[1:4],
    code(RUN_V5),
]

V6_CELLS = [
    md("""# v6: the matcher learns from 100% of train entities

Needs on Drive: `ber_v3/parquet/`, `ber_v3/work_ids_ce10.parquet`, `ber_v3/ckpt/`. Run cells 1-4, then cell 5 (~1.3 h).
Before v5 has finished this gives v4 + all train data (`ber_v3/out_v6/`); after v5 has saved its LLM scores to the
checkpoint it gives v5 + all train data (`ber_v3/out_v6_llm/`). Log: `ber_v3/run_v6.log`.
"""),
    *RUN_CELLS[1:4],
    code(RUN_V6),
]


def main() -> None:
    write(CELLS, OUT)


def write(cells: list, out: Path) -> None:
    cells = list(cells)
    # The code cell defining CODE must run before the unpack cell, so it goes right after the Drive cell.
    cells.insert(2, code(f"CODE = '{code_blob()}'"))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["run"]:
        write(RUN_CELLS, ROOT / "colab" / "run_v4.ipynb")
    elif sys.argv[1:] == ["v5"]:
        write(V5_CELLS, ROOT / "colab" / "run_v5.ipynb")
    elif sys.argv[1:] == ["v6"]:
        write(V6_CELLS, ROOT / "colab" / "run_v6.ipynb")
    else:
        main()
