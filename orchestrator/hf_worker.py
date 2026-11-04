"""Muse Glimmer (Meta, open weights, Apache-2.0) as a headless worker, served with
transformers, with one LoRA adapter per tenant (PEFT multi-adapter)."""
from __future__ import annotations

import torch

from .workers import WorkRequest


class HFWorker:
    def __init__(self, model, tokenizer, name: str, max_new_tokens: int = 384, max_prompt_tokens: int = 1536):
        self.model, self.tok, self.name = model, tokenizer, name
        self.max_new_tokens, self.max_prompt_tokens = max_new_tokens, max_prompt_tokens
        self.tok.padding_side = "left"
        ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|eot|>", "<|end_of_text|>")]
        self.eos = [i for i in ids if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id]
        self.adapters: set[str] = set()

    def has_peft(self) -> bool:
        return hasattr(self.model, "peft_config")

    def set_adapter(self, adapter: str | None):
        if adapter is None:
            return
        if adapter not in self.adapters:
            raise KeyError(f"adapter {adapter!r} is not loaded")
        self.model.set_adapter(adapter)

    @torch.inference_mode()
    def run_batch(self, reqs: list[WorkRequest]) -> list[str]:
        adapters = {r.adapter for r in reqs}
        if len(adapters) != 1:
            raise ValueError("one adapter per batch (the orchestrator batches per tenant)")
        adapter = adapters.pop()
        prompts = [self.tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True) for r in reqs]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_prompt_tokens,
                       add_special_tokens=False).to(self.model.device)
        kw = dict(max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=self.eos or None,
                  pad_token_id=self.tok.pad_token_id)
        if adapter is None and self.has_peft():
            if self.model.active_adapter not in self.model.peft_config:
                self.model.set_adapter(next(iter(self.model.peft_config)))
            with self.model.disable_adapter():
                out = self.model.generate(**enc, **kw)
        else:
            self.set_adapter(adapter)
            out = self.model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(row, skip_special_tokens=False).replace(self.tok.pad_token or "\0", "") for row in new]
