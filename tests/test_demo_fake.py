import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_end_to_end_on_cpu(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "demo" / "run_demo.py"), "--backend", "fake", "--out", str(tmp_path),
                        "--n-traj", "60", "--n-bench", "60"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((tmp_path / "summary.json").read_text())
    ms = s["results"]["machine-shop"]["hatches"]
    assert [h["decision"] for h in ms] == ["rollback", "promote"]
    assert s["cross_tenant_file"].startswith("blocked")
