import shutil
import sqlite3

import pytest

from memory import MemoryStore, TraceIds, HashingEmbedder, TenantIsolationError

IDS = TraceIds("p", "j", "t", "u1")


def _two(tmp_path):
    e = HashingEmbedder()
    a = MemoryStore(tmp_path, "tenant-a", e)
    b = MemoryStore(tmp_path, "tenant-b", e)
    a.add("Spindle noise means code SPN for tenant A", source="a/handbook.md", ids=IDS, kind="rule")
    a.add("Secret pricing of tenant A: 1234 per day", source="a/pricing.md", ids=IDS, kind="rule")
    b.add("Tenant B books trainings by groups of ten", source="b/handbook.md", ids=IDS, kind="rule")
    return a, b


def test_tenant_never_reads_another_tenant(tmp_path):
    a, b = _two(tmp_path)
    for q in ["Spindle noise means code SPN for tenant A", "secret pricing 1234", "anything at all"]:
        hits = b.search(q, k=10)
        assert hits, "tenant B has its own memory"
        assert all(h.tenant_id == "tenant-b" for h in hits)
        assert all("tenant A" not in h.text for h in hits)
    assert {m.source for m in b.all()} == {"b/handbook.md"}
    assert a.path != b.path


def test_every_memory_carries_source_and_four_ids(tmp_path):
    a, _ = _two(tmp_path)
    for m in a.all():
        assert m.source
        assert m.ids.project_id and m.ids.job_id and m.ids.task_id and m.ids.run_uid
    with pytest.raises(ValueError):
        TraceIds("p", "j", "", "u")
    with pytest.raises(ValueError):
        a.add("x", source="", ids=IDS)
    with pytest.raises(ValueError):
        a.add("x", source="s", ids={"project_id": "p"})  # not a TraceIds


@pytest.mark.parametrize("bad", ["../tenant-a", "tenant-a/../x", "A", "", "x", "tenant a", "/etc"])
def test_tenant_id_cannot_escape(tmp_path, bad):
    with pytest.raises(ValueError):
        MemoryStore(tmp_path, bad, HashingEmbedder())


def test_copied_store_is_refused(tmp_path):
    a, _ = _two(tmp_path)
    (tmp_path / "tenant-c").mkdir()
    shutil.copy(a.path, tmp_path / "tenant-c" / "memory.sqlite")
    with pytest.raises(TenantIsolationError):
        MemoryStore(tmp_path, "tenant-c", HashingEmbedder())


def test_tampered_row_is_refused(tmp_path):
    a, b = _two(tmp_path)
    db = sqlite3.connect(b.path)
    db.execute("UPDATE memory SET tenant_id='tenant-a'")
    db.commit()
    with pytest.raises(TenantIsolationError):
        b.search("anything", k=3)
