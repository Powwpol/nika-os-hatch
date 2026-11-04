import pytest

from demo.tenants import TENANTS, make_tasks, check
from memory import MemoryStore, TraceIds, HashingEmbedder, TenantIsolationError
from orchestrator import Orchestrator, TrajectoryStore, FakeWorker, parse_answer


def _orch(tmp_path, tid="machine-shop"):
    ms = MemoryStore(tmp_path, tid, HashingEmbedder())
    for src, text in TENANTS[tid]["handbook"]:
        ms.add(text, source=src, ids=TraceIds("p", "ingest", "t", "u"), kind="rule")
    ts = TrajectoryStore(tmp_path, tid)
    return Orchestrator(ms, ts, TENANTS[tid]["display"], lambda e, a: check(tid, e, a), k=3, batch_size=4), ms, ts


def test_trajectory_is_recorded_with_prompt_tools_output_verdict_ids(tmp_path):
    orch, ms, ts = _orch(tmp_path)
    tasks = make_tasks("machine-shop", 10, seed=1)
    worker = FakeWorker({t.task_id: t.expected for t in tasks}, accuracy=0.5)
    ids = TraceIds("proj", "collect", "task-7", "abc123")
    out = orch.run(tasks, [worker, worker], ids)
    stored = ts.all()
    assert len(stored) == len(out) == 10
    t = stored[0]
    assert t.messages[0]["role"] == "system" and t.messages[1]["content"] == tasks[0].text
    assert t.tool_calls[0]["name"] == "memory.search" and len(t.tool_calls[0]["result"]) == 3
    assert all("memory_id" in h and "source" in h for h in t.tool_calls[0]["result"])
    assert t.tool_calls[-1]["name"] == "answer.validate"
    assert t.raw_output and isinstance(t.answer, dict)
    assert t.verdict == check("machine-shop", tasks[0].expected, t.answer)[0]
    assert t.ids == ids
    assert 0 < sum(x.verdict for x in stored) < 10
    # every trajectory is also ingested into the tenant's memory with its ids
    assert len(ms.all(kind="trajectory")) == 10


def test_trajectory_store_refuses_other_tenant(tmp_path):
    orch, ms, ts = _orch(tmp_path)
    tasks = make_tasks("consulting-firm", 2, seed=1)
    worker = FakeWorker({t.task_id: t.expected for t in tasks})
    with pytest.raises(ValueError):
        orch.run(tasks, [worker], TraceIds("p", "j", "t", "u"))


def test_parse_answer_glimmer_channels():
    raw = ' to=self<|message|>The code is {"x": 1} maybe<|eom|><|start|>assistant to=user<|message|>{"code": "SPN-1", "priority": "P1", "asset": "CNC-01"}<|eot|>'
    assert parse_answer(raw) == {"code": "SPN-1", "priority": "P1", "asset": "CNC-01"}
    assert parse_answer("no json here") is None
