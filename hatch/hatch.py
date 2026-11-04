"""The hatching loop.

trigger (measured bottleneck) -> pre-registered bench (written before training)
-> LoRA training on the tenant's own trajectories -> paired evaluation
-> promote (adapter becomes the tenant's active adapter) or roll back (adapter is
discarded, the tenant keeps what it had). Every step is journaled.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Protocol

from .bench import decide, load_preregistration
from .registry import AdapterRegistry, HatchJournal


class Trainer(Protocol):
    def train(self, examples: list[dict], name: str, out_dir: Path, log: Callable[[str], None]) -> dict: ...
    def discard(self, name: str) -> None: ...


def bottleneck(trajs, min_trajectories: int = 50, max_fail_rate: float = 0.2) -> dict:
    n = len(trajs)
    fails = sum(1 for t in trajs if not t.verdict)
    rate = fails / n if n else 0.0
    return {"n": n, "fails": fails, "fail_rate": rate, "min_trajectories": min_trajectories,
            "max_fail_rate": max_fail_rate, "hatch": n >= min_trajectories and rate > max_fail_rate}


def sft_examples(trajs, shuffle_targets: bool = False, seed: int = 0) -> list[dict]:
    """One example per trajectory: the exact prompt the worker saw, and as target the
    answer the tenant accepted (the worker's answer when it passed, the tenant's
    correction when it failed)."""
    ex = [{"messages": t.messages, "target": json.dumps(t.expected, ensure_ascii=False), "task_id": t.task_id,
           "from": "accepted" if t.verdict else "correction"} for t in trajs if t.expected is not None]
    if shuffle_targets:  # negative control: break the prompt -> answer link
        targets = [e["target"] for e in ex]
        random.Random(seed).shuffle(targets)
        for e, tg in zip(ex, targets):
            e["target"], e["from"] = tg, "shuffled"
    return ex


class Hatcher:
    def __init__(self, root: str | Path, tenant_id: str, trainer: Trainer,
                 evaluate: Callable[[list, str | None], dict[str, bool]], log: Callable[[str], None] = print):
        self.root, self.tenant_id = Path(root), tenant_id
        self.trainer, self.evaluate, self.log = trainer, evaluate, log
        self.registry = AdapterRegistry(root, tenant_id)
        self.journal = HatchJournal(root, tenant_id)
        self._base_cache: dict[str, bool] | None = None

    def base_results(self, bench_tasks) -> dict[str, bool]:
        if self._base_cache is None:
            self._base_cache = self.evaluate(bench_tasks, self.registry.active)
        return self._base_cache

    def hatch(self, trajs, bench_tasks, prereg_path: Path, name: str, control: bool = False,
              train_kwargs: dict | None = None) -> dict:
        prereg = load_preregistration(prereg_path)  # refuses if the protocol changed
        if prereg["protocol"]["tenant_id"] != self.tenant_id:
            raise RuntimeError("pre-registration belongs to another tenant")
        bench_ids = {it["task_id"] for it in prereg["protocol"]["items"]}
        if any(t.task_id in bench_ids for t in trajs):
            raise RuntimeError("bench items leaked into the training trajectories")
        for t in trajs:
            if t.tenant_id != self.tenant_id:
                raise RuntimeError("trajectory from another tenant in the hatch set")
        ex = sft_examples(trajs, shuffle_targets=control, seed=prereg["protocol"]["seed"])
        self.journal.write("train_start", adapter=name, control=control, n_examples=len(ex),
                           n_corrections=sum(e["from"] == "correction" for e in ex),
                           prereg_sha256=prereg["sha256"], **(train_kwargs or {}))
        out_dir = self.registry.candidate_dir(name)
        stats = self.trainer.train(ex, name, out_dir, self.log)
        self.journal.write("train_end", adapter=name, **stats)
        base = self.base_results(bench_tasks)
        cand = self.evaluate(bench_tasks, name)
        v = decide(prereg_path, base, cand)
        vd = v.as_dict()
        if v.promote:
            self.registry.promote(name, vd)
            self.journal.write("promote", adapter=name, control=control, **vd)
            self.log(f"[{self.tenant_id}] PROMOTE {name}: {v.base_acc:.3f} -> {v.cand_acc:.3f} "
                     f"(wins {v.wins}, losses {v.losses}, p={v.p_sign:.2g}, boot_lo={v.boot_lower:+.3f})")
        else:
            self.trainer.discard(name)
            self.registry.rollback(name, vd)
            self.journal.write("rollback", adapter=name, control=control, **vd)
            self.log(f"[{self.tenant_id}] ROLLBACK {name}: {v.base_acc:.3f} -> {v.cand_acc:.3f} "
                     f"(wins {v.wins}, losses {v.losses}, p={v.p_sign:.2g}, boot_lo={v.boot_lower:+.3f})")
        return {"adapter": name, "control": control, "train": stats, "verdict": vd,
                "decision": "promote" if v.promote else "rollback"}
