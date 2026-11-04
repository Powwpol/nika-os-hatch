"""Orchestrator: plans each task, calls tools on the worker's behalf, dispatches to
headless workers, verifies, and records one trajectory per task in the tenant's frame.

Per task the trajectory holds: the exact prompt (messages), the tool calls made
(memory retrieval, answer validation), the worker's raw output, the parsed
answer, the verdict of the tenant's checker and the four trace ids.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from memory.store import MemoryStore, TraceIds
from .trajectories import Trajectory, TrajectoryStore
from .workers import WorkRequest

_JSON = re.compile(r"\{[^{}]*\}", re.S)


def build_messages(display: str, rules: list[str], task_text: str, reasoning: str = "low") -> list[dict]:
    mem = "\n".join(f"- {r}" for r in rules)
    system = (f"You are the agent of a {display}. Use the tenant memory below.\n"
              f"Tenant memory (retrieved):\n{mem}\n\n"
              "Reply with exactly one JSON object and nothing else.\n"
              f"Reasoning strength: {reasoning}.")
    return [{"role": "system", "content": system}, {"role": "user", "content": task_text}]


def parse_answer(raw: str) -> dict | None:
    """Take the user-channel message if present, then the last JSON object in it."""
    text = raw
    for marker in ("to=user<|message|>", "<|message|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            break
    for end in ("<|eot|>", "<|eom|>", "<|end_of_text|>"):
        text = text.split(end, 1)[0]
    found = _JSON.findall(text) or _JSON.findall(raw)
    for cand in reversed(found):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


class Orchestrator:
    def __init__(self, memory: MemoryStore, trajectories: TrajectoryStore, display: str,
                 checker: Callable[[dict, dict | None], tuple[bool, dict]], k: int = 4, batch_size: int = 16,
                 n_workers: int = 1):
        if memory.tenant_id != trajectories.tenant_id:
            raise ValueError("memory and trajectory stores must belong to the same tenant")
        self.tenant_id = memory.tenant_id
        self.memory, self.traj, self.display = memory, trajectories, display
        self.checker, self.k, self.batch_size, self.n_workers = checker, k, batch_size, n_workers

    def plan(self, task) -> tuple[list[dict], list[dict]]:
        """Pinned memories (e.g. the answer format) always go in; then the top-k retrieved ones."""
        pinned = [m for m in self.memory.all(kind="rule") if m.meta.get("pinned")]
        hits = [h for h in self.memory.search(task.text, k=self.k + len(pinned), kind="rule")
                if not h.meta.get("pinned")][:self.k]
        tool_calls = [{"name": "memory.search", "args": {"query": task.text, "k": self.k, "kind": "rule"},
                       "result": [{"memory_id": h.id, "source": h.source, "score": round(h.score, 4)} for h in hits]
                       + [{"memory_id": m.id, "source": m.source, "score": None, "pinned": True} for m in pinned]}]
        return build_messages(self.display, [h.text for h in hits] + [m.text for m in pinned], task.text), tool_calls

    def run(self, tasks, workers: list, ids: TraceIds, adapter: str | None = None, record: bool = True,
            log: Callable[[str], None] = lambda s: None) -> list[Trajectory]:
        for t in tasks:
            if t.tenant_id != self.tenant_id:
                raise ValueError(f"task {t.task_id} belongs to {t.tenant_id}, orchestrator serves {self.tenant_id}")
        planned = [(t, *self.plan(t)) for t in tasks]
        batches = [planned[i:i + self.batch_size] for i in range(0, len(planned), self.batch_size)]

        def dispatch(job):
            bi, batch = job
            worker = workers[bi % len(workers)]
            reqs = [WorkRequest(self.tenant_id, t.task_id, msgs, adapter) for t, msgs, _ in batch]
            t0 = time.time()
            outs = worker.run_batch(reqs)
            return bi, worker, outs, (time.time() - t0) / max(1, len(batch))

        results: dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(self.n_workers, len(workers)))) as ex:
            for bi, worker, outs, lat in ex.map(dispatch, list(enumerate(batches))):
                results[bi] = (worker, outs, lat)
                log(f"[{self.tenant_id}] batch {bi + 1}/{len(batches)} done by {worker.name} ({lat:.2f}s/task)")

        trajs = []
        for bi, batch in enumerate(batches):
            worker, outs, lat = results[bi]
            for (task, msgs, tools), raw in zip(batch, outs):
                ans = parse_answer(raw)
                ok, detail = self.checker(task.expected, ans)
                tools = tools + [{"name": "answer.validate", "args": {"json": ans is not None}, "result": detail}]
                tr = Trajectory(tenant_id=self.tenant_id, task_id=task.task_id, ids=ids, worker=worker.name,
                                adapter=adapter, messages=msgs, tool_calls=tools, raw_output=raw, answer=ans,
                                verdict=ok, verdict_detail=detail, expected=task.expected, latency_s=lat)
                if record:
                    self.traj.add(tr, commit=False)
                    self.memory.add(f"Task {task.task_id}: {task.text} -> verdict {'pass' if ok else 'fail'}",
                                    source=f"trajectory:{task.task_id}", ids=ids, kind="trajectory",
                                    meta={"verdict": ok, "adapter": adapter}, commit=False)
                trajs.append(tr)
        if record:
            self.traj.commit()
            self.memory.commit()
        return trajs
