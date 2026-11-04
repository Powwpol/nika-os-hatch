"""Per-tenant persistent memory on SQLite.

Isolation model (the "tenant frame"):

* one SQLite file per tenant, under ``<root>/<tenant_id>/memory.sqlite``;
* a ``MemoryStore`` is bound to exactly one tenant when it is opened; no read
  method takes a tenant argument, so a caller cannot ask for another tenant's rows;
* every row also stores its ``tenant_id``, and every read re-checks it
  (defence in depth: a copied or tampered file raises ``TenantIsolationError``);
* tenant ids are restricted to ``[a-z0-9-]`` so a tenant id can never escape
  its directory.

Every memory carries its ``source`` and four trace ids
(``project_id``, ``job_id``, ``task_id``, ``run_uid``). Writes without them are refused.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,62}$")


class TenantIsolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceIds:
    project_id: str
    job_id: str
    task_id: str
    run_uid: str

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"trace id '{k}' is required and must be a non-empty string")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Memory:
    id: int
    tenant_id: str
    text: str
    source: str
    kind: str
    ids: TraceIds
    created_at: float
    meta: dict = field(default_factory=dict)
    score: float | None = None


def validate_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not _TENANT_RE.match(tenant_id):
        raise ValueError(f"invalid tenant id: {tenant_id!r} (allowed: [a-z0-9-], 2-63 chars)")
    return tenant_id


SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant (tenant_id TEXT PRIMARY KEY, embedder TEXT NOT NULL, dim INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL CHECK (length(source) > 0),
  kind TEXT NOT NULL,
  project_id TEXT NOT NULL CHECK (length(project_id) > 0),
  job_id TEXT NOT NULL CHECK (length(job_id) > 0),
  task_id TEXT NOT NULL CHECK (length(task_id) > 0),
  run_uid TEXT NOT NULL CHECK (length(run_uid) > 0),
  created_at REAL NOT NULL,
  meta TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_kind ON memory(kind);
"""


class MemoryStore:
    def __init__(self, root: str | Path, tenant_id: str, embedder):
        self.tenant_id = validate_tenant_id(tenant_id)
        self.root = Path(root)
        self.dir = (self.root / self.tenant_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.dir.resolve().parent != self.root.resolve():
            raise TenantIsolationError("tenant directory escapes the memory root")
        self.path = self.dir / "memory.sqlite"
        self.embedder = embedder
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        dim = int(embedder.embed(["probe"]).shape[1])
        row = self.db.execute("SELECT tenant_id, embedder, dim FROM tenant").fetchone()
        if row is None:
            self.db.execute("INSERT INTO tenant VALUES (?,?,?)", (self.tenant_id, embedder.name, dim))
            self.db.commit()
        else:
            if row[0] != self.tenant_id:
                raise TenantIsolationError(f"store file belongs to tenant {row[0]!r}, not {self.tenant_id!r}")
            if row[1] != embedder.name or row[2] != dim:
                raise ValueError(f"store was built with embedder {row[1]} (dim {row[2]}), not {embedder.name} (dim {dim})")

    # ---------------------------------------------------------------- writes
    def add(self, text: str, *, source: str, ids: TraceIds, kind: str = "note", meta: dict | None = None,
            commit: bool = True) -> int:
        if not isinstance(ids, TraceIds):
            raise ValueError("ids must be a TraceIds with project_id, job_id, task_id and run_uid")
        if not source or not str(source).strip():
            raise ValueError("source is required")
        if not text or not text.strip():
            raise ValueError("text is required")
        vec = self.embedder.embed([text])[0].astype(np.float32)
        cur = self.db.execute(
            "INSERT INTO memory (tenant_id,text,source,kind,project_id,job_id,task_id,run_uid,created_at,meta,embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.tenant_id, text, source, kind, ids.project_id, ids.job_id, ids.task_id, ids.run_uid,
             time.time(), json.dumps(meta or {}, ensure_ascii=False), vec.tobytes()))
        if commit:
            self.db.commit()
        return int(cur.lastrowid)

    # ----------------------------------------------------------------- reads
    def _row(self, r, score=None) -> Memory:
        if r[1] != self.tenant_id:
            raise TenantIsolationError(f"row {r[0]} belongs to tenant {r[1]!r}")
        return Memory(id=r[0], tenant_id=r[1], text=r[2], source=r[3], kind=r[4],
                      ids=TraceIds(r[5], r[6], r[7], r[8]), created_at=r[9], meta=json.loads(r[10]), score=score)

    _COLS = "id,tenant_id,text,source,kind,project_id,job_id,task_id,run_uid,created_at,meta"

    def all(self, kind: str | None = None) -> list[Memory]:
        q = f"SELECT {self._COLS} FROM memory" + (" WHERE kind=?" if kind else "") + " ORDER BY id"
        return [self._row(r) for r in self.db.execute(q, (kind,) if kind else ())]

    def count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM memory").fetchone()[0])

    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[Memory]:
        q = f"SELECT {self._COLS}, embedding FROM memory" + (" WHERE kind=?" if kind else "")
        rows = list(self.db.execute(q, (kind,) if kind else ()))
        if not rows:
            return []
        mat = np.stack([np.frombuffer(r[-1], dtype=np.float32) for r in rows])
        qv = self.embedder.embed([query])[0]
        scores = mat @ qv
        order = np.argsort(-scores)[:k]
        return [self._row(rows[i][:-1], float(scores[i])) for i in order]

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()
