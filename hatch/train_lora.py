"""LoRA training (PEFT) on an already loaded Meta open-weight model, one adapter per
tenant. Loss only on the answer tokens; plain AdamW loop, bf16, gradient checkpointing."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"


def encode_example(tok, messages, target, max_len=1792):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": target}], tokenize=False)
    if not full.startswith(prompt):
        raise RuntimeError("chat template: full conversation does not start with the generation prompt")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        raise RuntimeError("tokenization of the prompt is not a prefix of the full conversation")
    f_ids = f_ids[:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels


class LoraTrainer:
    def __init__(self, model, tok, r=16, alpha=32, dropout=0.05, lr=2e-4, steps=200, batch=4, warmup=10,
                 seed=0, targets=LORA_TARGETS, grad_ckpt=True):
        self.model, self.tok = model, tok
        self.cfg = dict(r=r, alpha=alpha, dropout=dropout, lr=lr, steps=steps, batch=batch, warmup=warmup, seed=seed,
                        targets=targets)
        self.grad_ckpt = grad_ckpt
        self.loaded: set[str] = set()

    def _attach(self, name):
        from peft import LoraConfig, get_peft_model

        c = self.cfg
        lc = LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"], target_modules=c["targets"],
                        bias="none", task_type="CAUSAL_LM")
        if not hasattr(self.model, "peft_config"):
            self.model = get_peft_model(self.model, lc, adapter_name=name)
        else:
            self.model.add_adapter(name, lc)
        self.model.set_adapter(name)
        n_train = 0
        for n, p in self.model.named_parameters():
            p.requires_grad = ("lora_" in n and f".{name}." in n)
            n_train += p.numel() if p.requires_grad else 0
        self.loaded.add(name)
        return n_train

    def train(self, examples, name, out_dir: Path, log=print) -> dict:
        c = self.cfg
        torch.manual_seed(c["seed"])
        rng = random.Random(c["seed"])
        n_train = self._attach(name)
        m = self.model
        if self.grad_ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
        m.train()
        enc = [encode_example(self.tok, e["messages"], e["target"]) for e in examples]
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / c["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"]))))
        pad = self.tok.pad_token_id
        dev = next(p for p in m.parameters()).device
        order, losses, t0 = [], [], time.time()
        for step in range(c["steps"]):
            if len(order) < c["batch"]:
                idx = list(range(len(enc)))
                rng.shuffle(idx)
                order += idx
            batch = [enc[order.pop()] for _ in range(c["batch"])]
            L = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in batch], device=dev)
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device=dev)
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device=dev)
            out = m(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss))
            if step % 10 == 0 or step == c["steps"] - 1:
                log(f"  [{name}] step {step + 1}/{c['steps']} loss {losses[-1]:.4f} ({time.time() - t0:.0f}s)")
        m.eval()
        if self.grad_ckpt:
            m.gradient_checkpointing_disable()
        out_dir.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(str(out_dir), selected_adapters=[name])
        k = max(1, len(losses) // 10)
        return {"steps": c["steps"], "batch": c["batch"], "lr": c["lr"], "lora_r": c["r"], "trainable_params": n_train,
                "loss_first10": sum(losses[:k]) / k, "loss_last10": sum(losses[-k:]) / k,
                "train_seconds": round(time.time() - t0, 1), "n_examples": len(enc)}

    def discard(self, name):
        if name in self.loaded and hasattr(self.model, "delete_adapter"):
            self.model.delete_adapter(name)
            self.loaded.discard(name)
