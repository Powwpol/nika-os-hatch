"""Headless workers. A worker receives a batch of chat requests and returns raw text.

* ``FakeWorker``: deterministic, CPU-only, for tests and CI.
* ``HFWorker`` (orchestrator/hf_worker.py): a Meta open-weight model (Muse Glimmer)
  served with transformers, with one optional LoRA adapter per tenant.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass
class WorkRequest:
    tenant_id: str
    task_id: str
    messages: list
    adapter: str | None = None  # name of the tenant's active adapter, None = base model


class FakeWorker:
    """Answers correctly with probability ``accuracy`` (per adapter), else a wrong but well-formed answer."""

    name = "fake-worker"

    def __init__(self, oracle: dict[str, dict], accuracy: float = 0.5, adapter_accuracy: dict[str, float] | None = None,
                 seed: int = 0):
        self.oracle = oracle  # task_id -> expected
        self.accuracy = accuracy
        self.adapter_accuracy = adapter_accuracy or {}
        self.seed = seed

    def run_batch(self, reqs: list[WorkRequest]) -> list[str]:
        out = []
        for r in reqs:
            acc = self.adapter_accuracy.get(r.adapter, self.accuracy) if r.adapter else self.accuracy
            rng = random.Random(f"{self.seed}:{r.task_id}:{r.adapter}")
            exp = dict(self.oracle[r.task_id])
            if rng.random() >= acc:
                k = sorted(exp)[0]
                exp[k] = f"{exp[k]}X"
            out.append(f"thinking...<|message|>{json.dumps(exp)}")
        return out
