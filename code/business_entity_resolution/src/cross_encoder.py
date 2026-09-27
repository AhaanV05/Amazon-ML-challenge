"""Cross-encoder pair scorer. Runs on a GPU (Colab A100/H100); needs torch + transformers only.

Train:  python cross_encoder.py train --data DIR [--model xlm-roberta-base] [--out DIR/ce_model]
Score:  python cross_encoder.py score --data DIR --pairs work_pairs.parquet --records records_train.parquet
        python cross_encoder.py score --data DIR --pairs test_pairs.parquet --records records_test.parquet

DIR holds the files written by `python -m src.ce_data` on the laptop: records_*.parquet (key, text) and
*_pairs.parquet (q_rec, d_rec[, label]). Scores are written next to the pairs as *_scores.parquet
(q_rec, d_rec, ce_logit).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_pairs(data: str, pairs_file: str, records_file: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rec = pd.read_parquet(os.path.join(data, records_file))
    text = pd.Series(rec["text"].to_numpy(), index=rec["key"].to_numpy())
    pairs = pd.read_parquet(os.path.join(data, pairs_file))
    a = text.reindex(pairs["q_rec"].to_numpy()).fillna("").to_numpy()
    b = text.reindex(pairs["d_rec"].to_numpy()).fillna("").to_numpy()
    return pairs, a, b


class PairBatches(torch.utils.data.Dataset):
    """Each item is a whole tokenized batch, so tokenization runs in DataLoader workers."""

    def __init__(self, tok, a, b, y, batches, max_len):
        self.tok, self.a, self.b, self.y, self.batches, self.max_len = tok, a, b, y, batches, max_len

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        enc = self.tok(self.a[idx].tolist(), self.b[idx].tolist(), truncation="longest_first",
                       max_length=self.max_len, padding="longest", return_tensors="pt")
        enc["idx"] = torch.from_numpy(idx)
        if self.y is not None:
            enc["labels"] = torch.from_numpy(self.y[idx].astype(np.float32))
        return dict(enc)


def loader(tok, a, b, y, batches, max_len, workers):
    return torch.utils.data.DataLoader(PairBatches(tok, a, b, y, batches, max_len), batch_size=None,
                                       num_workers=workers, pin_memory=True, persistent_workers=False)


def length_batches(a, b, size):
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    return [order[i:i + size] for i in range(0, len(order), size)]


@torch.no_grad()
def predict(model, tok, a, b, args) -> np.ndarray:
    model.eval()
    out = np.empty(len(a), dtype=np.float32)
    dl = loader(tok, a, b, None, length_batches(a, b, args.infer_batch), args.max_len, args.workers)
    for batch in dl:
        idx = batch.pop("idx").numpy()
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.squeeze(-1)
        out[idx] = logits.float().cpu().numpy()
    return out


def train(args) -> None:
    pairs, a, b = load_pairs(args.data, "ce_pairs.parquet", "records_train.parquet")
    y = pairs["label"].to_numpy().astype(np.float32)
    held = (pd.util.hash_array(pairs["q_rec"].to_numpy()) % 100) < 3
    tr = np.flatnonzero(~held)
    if args.max_train_pairs and len(tr) > args.max_train_pairs:
        tr = np.random.default_rng(0).choice(tr, args.max_train_pairs, replace=False)
    va = np.flatnonzero(held)
    print(f"train pairs {len(tr):,} (positive {y[tr].mean():.1%}) | held-out {len(va):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        batches = [perm[i:i + args.batch] for i in range(0, len(perm), args.batch)]
        for batch in loader(tok, a, b, y, batches, args.max_len, args.workers):
            batch.pop("idx")
            labels = batch.pop("labels").cuda(non_blocking=True)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} | {rate:,.0f} pairs/s", flush=True)
    p = predict(model, tok, a[va], b[va], args)
    metrics = {"model": args.model, "train_pairs": int(len(tr)), "heldout_pairs": int(len(va)),
               "heldout_auc": float(roc_auc_score(y[va], p)),
               "heldout_logloss": float(log_loss(y[va], 1 / (1 + np.exp(-p)))), "minutes": (time.time() - t0) / 60}
    print(json.dumps(metrics, indent=2), flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def score(args) -> None:
    pairs, a, b = load_pairs(args.data, args.pairs, args.records)
    tok = AutoTokenizer.from_pretrained(args.out)
    model = AutoModelForSequenceClassification.from_pretrained(args.out).cuda()
    t0 = time.time()
    logit = predict(model, tok, a, b, args)
    out = pairs[["q_rec", "d_rec"]].assign(ce_logit=logit)
    path = os.path.join(args.data, args.pairs.replace("_pairs", "_scores"))
    out.to_parquet(path, index=False)
    print(f"scored {len(out):,} pairs in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", default="test_pairs.parquet")
    ap.add_argument("--records", default="records_test.parquet")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--infer-batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.data, "ce_model")
    torch.manual_seed(0)
    train(args) if args.mode == "train" else score(args)


if __name__ == "__main__":
    main()
