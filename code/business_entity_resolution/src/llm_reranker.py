"""LLM reranker for the hardest pairs: Qwen3-Reranker (Apache-2.0) fine-tuned with LoRA. GPU; torch, transformers, peft.

Same interface as cross_encoder.py, so the pipeline routes pairs to it the same way:
  train:  python llm_reranker.py train --data DIR --out DIR [--model Qwen/Qwen3-Reranker-4B]
          (DIR holds ce_pairs.parquet with labels and records_train.parquet)
  score:  python llm_reranker.py score --data DIR --out MODEL_DIR --pairs X_pairs.parquet --records records_Y.parquet
The score is logit(yes) - logit(no) at the answer position, written as ce_logit to X_scores.parquet.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from cross_encoder import length_batches, load_pairs

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = ("Is the Document the same business as the Query? Both are 'name | address' records; allow typos, "
            "transliteration, abbreviations and missing address parts.")
MAX_CHARS = 200


def prompts(a, b) -> list[str]:
    return [f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {x[:MAX_CHARS]}\n<Document>: {y[:MAX_CHARS]}{SUFFIX}"
            for x, y in zip(a, b)]


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    return tok, tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


def yes_no_logit(model, tok, texts, yes: int, no: int) -> torch.Tensor:
    enc = tok(texts, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**enc, logits_to_keep=1).logits[:, -1, :]
    return (out[:, yes] - out[:, no]).float()


@torch.no_grad()
def predict(model, tok, yes: int, no: int, a, b, batch: int) -> np.ndarray:
    model.eval()
    out = np.empty(len(a), dtype=np.float32)
    batches = length_batches(a, b, batch)
    t0 = time.time()
    for i, idx in enumerate(batches):
        out[idx] = yes_no_logit(model, tok, prompts(a[idx], b[idx]), yes, no).cpu().numpy()
        if (i + 1) % max(1, len(batches) // 20) == 0:
            done = sum(len(x) for x in batches[:i + 1])
            print(f"  scored {done:,}/{len(a):,} | {done / (time.time() - t0):,.0f} pairs/s", flush=True)
    return out


def train(args) -> None:
    pairs, a, b = load_pairs(args.data, "ce_pairs.parquet", "records_train.parquet")
    y = pairs["label"].to_numpy().astype(np.float32)
    held = (pd.util.hash_array(pairs["q_rec"].to_numpy()) % 100) < 3
    tr, va = np.flatnonzero(~held), np.flatnonzero(held)
    rng = np.random.default_rng(0)
    if args.max_train_pairs and len(tr) > args.max_train_pairs:
        tr = rng.choice(tr, args.max_train_pairs, replace=False)
    print(f"train pairs {len(tr):,} (positive {y[tr].mean():.1%}) | held-out {len(va):,}", flush=True)

    tok, yes, no = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # LoRA weights train in fp32; the frozen base stays bf16
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(tr) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    perm = rng.permutation(tr)
    model.train()
    t0 = time.time()
    for step, i in enumerate(range(0, len(perm), args.batch), 1):
        idx = perm[i:i + args.batch]
        z = yes_no_logit(model, tok, prompts(a[idx], b[idx]), yes, no)
        loss = loss_fn(z, torch.from_numpy(y[idx]).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} | {step * args.batch / (time.time() - t0):,.0f} pairs/s",
                  flush=True)
    p = predict(model, tok, yes, no, a[va], b[va], args.infer_batch)
    metrics = {"model": args.model, "train_pairs": int(len(tr)), "heldout_pairs": int(len(va)),
               "heldout_auc": float(roc_auc_score(y[va], p)) if 0 < y[va].sum() < len(va) else None,
               "heldout_logloss": float(log_loss(y[va], 1 / (1 + np.exp(-p)), labels=[0, 1])),
               "minutes": (time.time() - t0) / 60}
    print(json.dumps(metrics, indent=2), flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def score(args) -> None:
    with open(os.path.join(args.out, "metrics.json")) as f:
        base_name = json.load(f)["model"]
    tok, yes, no = load_tokenizer(args.out)
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.out).merge_and_unload().cuda()
    pairs, a, b = load_pairs(args.data, args.pairs, args.records)
    t0 = time.time()
    out = pairs[["q_rec", "d_rec"]].assign(ce_logit=predict(model, tok, yes, no, a, b, args.infer_batch))
    path = os.path.join(args.data, args.pairs.replace("_pairs", "_scores"))
    out.to_parquet(path, index=False)
    print(f"scored {len(out):,} pairs in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="test_pairs.parquet")
    ap.add_argument("--records", default="records_test.parquet")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--infer-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)  # accepted for interface parity; tokenization is in-process
    args = ap.parse_args()
    train(args) if args.mode == "train" else score(args)


if __name__ == "__main__":
    main()
