"""Pre-registered paired benchmark.

The protocol (items, metric, test, alpha, decision rule, seed) is written and hashed
*before* any training. Evaluation re-hashes it and refuses to decide if it changed.
Statistics make no normality assumption:
  * exact one-sided sign test on discordant pairs (equivalent to an exact McNemar test);
  * paired bootstrap of the mean per-item difference (percentile interval).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

DECISION_RULE = ("promote iff (delta > 0) and (sign-test one-sided p < alpha) and "
                 "(bootstrap one-sided lower bound of delta at level 1-alpha > 0); otherwise roll back")


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def preregister(path: Path, tenant_id: str, items, model_id: str, alpha: float = 0.05, n_boot: int = 10000,
                seed: int = 20261104, extra: dict | None = None) -> dict:
    proto = {
        "tenant_id": tenant_id,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "metric": "per-item exact match of every expected field (binary)",
        "test": "exact one-sided sign test on discordant pairs + paired bootstrap of mean difference",
        "alpha": alpha, "n_boot": n_boot, "seed": seed,
        "decision_rule": DECISION_RULE,
        "items": [{"task_id": t.task_id, "text_sha256": hashlib.sha256(t.text.encode()).hexdigest(),
                   "expected": t.expected} for t in items],
        **(extra or {}),
    }
    proto["items_sha256"] = canonical_sha(proto["items"])
    sha = canonical_sha(proto)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"protocol": proto, "sha256": sha}, indent=1, ensure_ascii=False))
    return {"path": str(path), "sha256": sha, "n_items": len(items)}


def load_preregistration(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if canonical_sha(d["protocol"]) != d["sha256"]:
        raise RuntimeError("pre-registration was modified after registration: refusing to decide")
    return d


def sign_test_one_sided(wins: int, losses: int) -> float:
    """P(X >= wins) for X ~ Binomial(wins + losses, 1/2). Ties are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def paired_bootstrap(diffs: list[int], n_boot: int, seed: int, alpha: float) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(math.floor(alpha * n_boot))]
    lo2 = means[int(math.floor(alpha / 2 * n_boot))]
    hi2 = means[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return {"lower_one_sided": lo, "ci_two_sided": [lo2, hi2]}


@dataclass
class Verdict:
    n: int
    base_acc: float
    cand_acc: float
    delta: float
    wins: int
    losses: int
    ties: int
    p_sign: float
    boot_lower: float
    boot_ci: list
    promote: bool
    prereg_sha256: str

    def as_dict(self):
        return asdict(self)


def decide(prereg_path: Path, base_pass: dict[str, bool], cand_pass: dict[str, bool]) -> Verdict:
    d = load_preregistration(prereg_path)
    p = d["protocol"]
    ids = [it["task_id"] for it in p["items"]]
    missing = [i for i in ids if i not in base_pass or i not in cand_pass]
    if missing:
        raise RuntimeError(f"{len(missing)} pre-registered items were not evaluated (e.g. {missing[:3]})")
    b = [int(base_pass[i]) for i in ids]
    c = [int(cand_pass[i]) for i in ids]
    diffs = [ci - bi for bi, ci in zip(b, c)]
    wins, losses = diffs.count(1), diffs.count(-1)
    p_sign = sign_test_one_sided(wins, losses)
    boot = paired_bootstrap(diffs, p["n_boot"], p["seed"], p["alpha"])
    delta = sum(diffs) / len(diffs)
    promote = delta > 0 and p_sign < p["alpha"] and boot["lower_one_sided"] > 0
    return Verdict(n=len(ids), base_acc=sum(b) / len(b), cand_acc=sum(c) / len(c), delta=delta, wins=wins,
                   losses=losses, ties=len(ids) - wins - losses, p_sign=p_sign, boot_lower=boot["lower_one_sided"],
                   boot_ci=boot["ci_two_sided"], promote=promote, prereg_sha256=d["sha256"])
