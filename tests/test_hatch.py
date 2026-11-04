import json

import pytest

from demo.run_demo import FakeTrainer
from demo.tenants import TENANTS, make_tasks, check
from hatch.bench import preregister, sign_test_one_sided, decide
from hatch.hatch import Hatcher, bottleneck
from hatch.registry import AdapterRegistry
from memory import MemoryStore, TraceIds, HashingEmbedder
from orchestrator import Orchestrator, TrajectoryStore, FakeWorker

TID = "consulting-firm"


@pytest.fixture
def setup(tmp_path):
    ms = MemoryStore(tmp_path, TID, HashingEmbedder())
    for src, text in TENANTS[TID]["handbook"]:
        ms.add(text, source=src, ids=TraceIds("p", "ingest", "t", "u"), kind="rule")
    ts = TrajectoryStore(tmp_path, TID)
    orch = Orchestrator(ms, ts, "x", lambda e, a: check(TID, e, a), k=3, batch_size=16)
    collect = make_tasks(TID, 60, seed=1)
    bench = make_tasks(TID, 60, seed=2, exclude_texts={t.text for t in collect}, prefix="bench-")
    worker = FakeWorker({t.task_id: t.expected for t in collect + bench}, accuracy=0.5)
    trajs = orch.run(collect, [worker], TraceIds("p", "collect", "t", "u"))

    def evaluate(tasks, adapter):
        return {t.task_id: t.verdict for t in orch.run(tasks, [worker], TraceIds("p", "bench", "t", "u"), adapter=adapter)}

    pre = tmp_path / TID / "prereg.json"
    preregister(pre, TID, bench, "fake")
    hatcher = Hatcher(tmp_path, TID, FakeTrainer(worker), evaluate, log=lambda s: None)
    hatcher.collect_tasks = collect
    return tmp_path, hatcher, trajs, bench, pre


def test_sign_test_exact():
    assert sign_test_one_sided(0, 0) == 1.0
    assert sign_test_one_sided(5, 0) == pytest.approx(1 / 32)
    assert sign_test_one_sided(3, 3) == pytest.approx(42 / 64)


def test_bottleneck_trigger(setup):
    _, _, trajs, _, _ = setup
    assert bottleneck(trajs, min_trajectories=50, max_fail_rate=0.2)["hatch"]
    assert not bottleneck(trajs[:10], min_trajectories=50)["hatch"]


def test_promotion_on_gain(setup):
    root, hatcher, trajs, bench, pre = setup
    r = hatcher.hatch(trajs, bench, pre, "cf-v1")
    assert r["decision"] == "promote"
    reg = AdapterRegistry(root, TID)
    assert reg.active == "cf-v1" and (reg.dir / "promoted" / "cf-v1").exists()
    events = [e["event"] for e in hatcher.journal.read()]
    assert events == ["train_start", "train_end", "promote"]


def test_rollback_on_negative_control(setup):
    root, hatcher, trajs, bench, pre = setup
    r = hatcher.hatch(trajs, bench, pre, "cf-control", control=True)
    assert r["decision"] == "rollback"
    reg = AdapterRegistry(root, TID)
    assert reg.active is None and (reg.dir / "rolled_back" / "cf-control").exists()
    assert reg.history()[-1]["event"] == "rollback"


def test_rollback_keeps_previous_active(setup):
    root, hatcher, trajs, bench, pre = setup
    hatcher.hatch(trajs, bench, pre, "cf-v1")
    hatcher._base_cache = None
    hatcher.hatch(trajs, bench, pre, "cf-bad", control=True)
    assert AdapterRegistry(root, TID).active == "cf-v1"


def test_tampered_preregistration_is_refused(setup):
    _, hatcher, trajs, bench, pre = setup
    d = json.loads(pre.read_text())
    d["protocol"]["alpha"] = 0.5
    pre.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="modified"):
        hatcher.hatch(trajs, bench, pre, "cf-v1")


def test_bench_leak_is_refused(setup, tmp_path):
    _, hatcher, trajs, bench, _ = setup
    leak = tmp_path / "leak.json"
    preregister(leak, TID, [hatcher.collect_tasks[0]], "fake")  # a training item in the bench
    with pytest.raises(RuntimeError, match="leaked"):
        hatcher.hatch(trajs, bench, leak, "cf-v1")


def test_decide_no_gain_no_promotion(setup, tmp_path):
    _, _, _, bench, pre = setup
    same = {t.task_id: (i % 2 == 0) for i, t in enumerate(bench)}
    v = decide(pre, same, same)
    assert not v.promote and v.delta == 0
