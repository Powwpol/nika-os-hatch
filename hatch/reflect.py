"""Level 1 of evolution: prompt evolution (GEPA-style reflection, Agrawal et al. 2025, simplified).

The verifier's per-field failures over a tenant's trajectories are read back; for every
field that fails too often, the reflector looks in the tenant's own memory for the rules
that talk about that field and proposes to *pin* them into the worker instruction.
Output = a candidate instruction change, journaled. In this repository it is proposed and
recorded, not yet measured on the GPU bench (levels 2 and 3 are).
"""
from __future__ import annotations


def field_failures(trajs) -> dict[str, float]:
    n = len(trajs) or 1
    fails: dict[str, int] = {}
    for t in trajs:
        for k, ok in (t.verdict_detail.get("fields") or {}).items():
            fails[k] = fails.get(k, 0) + (0 if ok else 1)
    return {k: v / n for k, v in sorted(fails.items())}


def propose_pins(trajs, memory, threshold: float = 0.2) -> dict:
    rates = field_failures(trajs)
    rules = memory.all(kind="rule")
    proposals = []
    for field, rate in rates.items():
        if rate <= threshold:
            continue
        ranked = memory.search(f"the {field} is", k=len(rules), kind="rule")
        hits = [m for m in ranked if field.lower() in m.text.lower() and not m.meta.get("pinned")]
        proposals.append({"field": field, "fail_rate": round(rate, 3),
                          "pin": [{"memory_id": m.id, "source": m.source} for m in hits[:2]]})
    return {"field_fail_rates": {k: round(v, 3) for k, v in rates.items()}, "threshold": threshold,
            "proposals": proposals, "status": "proposed, not evaluated in this run"}
