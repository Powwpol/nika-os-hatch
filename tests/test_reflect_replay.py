import json
import subprocess
import sys
from pathlib import Path

from demo.tenants import TENANTS, make_tasks, check
from hatch.reflect import propose_pins
from memory import MemoryStore, TraceIds, HashingEmbedder
from orchestrator import Orchestrator, TrajectoryStore, FakeWorker

ROOT = Path(__file__).resolve().parents[1]


def test_reflector_pins_the_rule_of_the_failing_field(tmp_path):
    tid = "consulting-firm"
    ms = MemoryStore(tmp_path, tid, HashingEmbedder())
    for src, text in TENANTS[tid]["handbook"]:
        ms.add(text, source=src, ids=TraceIds("p", "j", "t", "u"), kind="rule", meta={"pinned": src.endswith("format.md")})
    orch = Orchestrator(ms, TrajectoryStore(tmp_path, tid), "x", lambda e, a: check(tid, e, a))
    tasks = make_tasks(tid, 40, seed=1)
    trajs = orch.run(tasks, [FakeWorker({t.task_id: t.expected for t in tasks}, accuracy=0.0)], TraceIds("p", "j", "t", "u"))
    # FakeWorker corrupts the first field alphabetically: 'days'
    rp = propose_pins(trajs, ms)
    assert rp["field_fail_rates"]["days"] == 1.0
    assert rp["proposals"][0]["field"] == "days"
    assert all("days" in s["source"] for s in rp["proposals"][0]["pin"])


def test_replay_reproduces_the_recorded_gpu_verdicts(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "demo" / "run_demo.py"), "--backend", "replay", "--out", str(tmp_path),
                        "--n-traj", "240", "--n-bench", "100", "--gen-batch", "48"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads((tmp_path / "summary.json").read_text())["results"]
    real = json.loads((ROOT / "demo" / "evidence" / "real_run" / "summary.json").read_text())["results"]
    for tid in real:
        for g, w in zip(got[tid]["hatches"], real[tid]["hatches"]):
            assert g["decision"] == w["decision"]
            for k in ("wins", "losses", "base_acc", "cand_acc", "p_sign"):
                assert g["verdict"][k] == w["verdict"][k]
