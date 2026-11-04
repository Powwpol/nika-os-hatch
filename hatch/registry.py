"""Per-tenant adapter registry and hatch journal (promotion / rollback)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from memory.store import validate_tenant_id


class AdapterRegistry:
    def __init__(self, root: str | Path, tenant_id: str):
        self.tenant_id = validate_tenant_id(tenant_id)
        self.dir = Path(root) / self.tenant_id / "adapters"
        for sub in ("candidates", "promoted", "rolled_back"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "registry.json"
        if not self.file.exists():
            self._write({"tenant_id": self.tenant_id, "active": None, "history": []})

    def _read(self) -> dict:
        d = json.loads(self.file.read_text())
        if d["tenant_id"] != self.tenant_id:
            raise RuntimeError("registry belongs to another tenant")
        return d

    def _write(self, d: dict):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.file)

    @property
    def active(self) -> str | None:
        return self._read()["active"]

    def active_path(self) -> Path | None:
        a = self.active
        return (self.dir / "promoted" / a) if a else None

    def candidate_dir(self, name: str) -> Path:
        return self.dir / "candidates" / name

    def promote(self, name: str, verdict: dict):
        d = self._read()
        src, dst = self.candidate_dir(name), self.dir / "promoted" / name
        if src.exists():
            shutil.move(str(src), str(dst))
        d["history"].append({"ts": time.time(), "event": "promote", "adapter": name, "previous": d["active"],
                             "verdict": verdict})
        d["active"] = name
        self._write(d)

    def rollback(self, name: str, verdict: dict):
        d = self._read()
        src, dst = self.candidate_dir(name), self.dir / "rolled_back" / name
        if src.exists():
            shutil.move(str(src), str(dst))
        d["history"].append({"ts": time.time(), "event": "rollback", "adapter": name, "kept": d["active"],
                             "verdict": verdict})
        self._write(d)

    def history(self) -> list:
        return self._read()["history"]


class HatchJournal:
    def __init__(self, root: str | Path, tenant_id: str):
        self.tenant_id = validate_tenant_id(tenant_id)
        self.path = Path(root) / self.tenant_id / "hatch_journal.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **data):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tenant_id": self.tenant_id,
               "event": event, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
