"""Per-tenant trajectory log (SQLite), same tenant frame as the memory store."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from memory.store import TraceIds, TenantIsolationError, validate_tenant_id


@dataclass
class Trajectory:
    tenant_id: str
    task_id: str
    ids: TraceIds
    worker: str
    adapter: str | None
    messages: list
    tool_calls: list
    raw_output: str
    answer: dict | None
    verdict: bool
    verdict_detail: dict
    expected: dict | None = None  # the tenant's correction, used for hatching
    latency_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    id: int | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant (tenant_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS trajectory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL, task_id TEXT NOT NULL,
  project_id TEXT NOT NULL, job_id TEXT NOT NULL, run_task_id TEXT NOT NULL, run_uid TEXT NOT NULL,
  worker TEXT NOT NULL, adapter TEXT,
  messages TEXT NOT NULL, tool_calls TEXT NOT NULL, raw_output TEXT NOT NULL,
  answer TEXT, verdict INTEGER NOT NULL, verdict_detail TEXT NOT NULL, expected TEXT,
  latency_s REAL NOT NULL, created_at REAL NOT NULL
);
"""


class TrajectoryStore:
    def __init__(self, root: str | Path, tenant_id: str):
        self.tenant_id = validate_tenant_id(tenant_id)
        d = Path(root) / self.tenant_id
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / "trajectories.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        row = self.db.execute("SELECT tenant_id FROM tenant").fetchone()
        if row is None:
            self.db.execute("INSERT INTO tenant VALUES (?)", (self.tenant_id,))
            self.db.commit()
        elif row[0] != self.tenant_id:
            raise TenantIsolationError(f"trajectory file belongs to {row[0]!r}")

    def add(self, t: Trajectory, commit: bool = True) -> int:
        if t.tenant_id != self.tenant_id:
            raise TenantIsolationError(f"trajectory for {t.tenant_id!r} cannot be written in {self.tenant_id!r}")
        i = t.ids
        cur = self.db.execute(
            "INSERT INTO trajectory (tenant_id,task_id,project_id,job_id,run_task_id,run_uid,worker,adapter,messages,"
            "tool_calls,raw_output,answer,verdict,verdict_detail,expected,latency_s,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.tenant_id, t.task_id, i.project_id, i.job_id, i.task_id, i.run_uid, t.worker, t.adapter,
             json.dumps(t.messages, ensure_ascii=False), json.dumps(t.tool_calls, ensure_ascii=False), t.raw_output,
             json.dumps(t.answer, ensure_ascii=False) if t.answer is not None else None, int(bool(t.verdict)),
             json.dumps(t.verdict_detail), json.dumps(t.expected, ensure_ascii=False) if t.expected is not None else None,
             float(t.latency_s), t.created_at))
        if commit:
            self.db.commit()
        t.id = int(cur.lastrowid)
        return t.id

    def all(self, job_id: str | None = None) -> list[Trajectory]:
        q = "SELECT * FROM trajectory" + (" WHERE job_id=?" if job_id else "") + " ORDER BY id"
        out = []
        for r in self.db.execute(q, (job_id,) if job_id else ()):
            if r[1] != self.tenant_id:
                raise TenantIsolationError(f"row {r[0]} belongs to {r[1]!r}")
            out.append(Trajectory(tenant_id=r[1], task_id=r[2], ids=TraceIds(r[3], r[4], r[5], r[6]), worker=r[7],
                                  adapter=r[8], messages=json.loads(r[9]), tool_calls=json.loads(r[10]),
                                  raw_output=r[11], answer=json.loads(r[12]) if r[12] else None, verdict=bool(r[13]),
                                  verdict_detail=json.loads(r[14]), expected=json.loads(r[15]) if r[15] else None,
                                  latency_s=r[16], created_at=r[17], id=r[0]))
        return out

    def commit(self):
        self.db.commit()

    def count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM trajectory").fetchone()[0])
