#!/usr/bin/env python3
"""End-to-end demo: per-tenant memory -> orchestrated headless workers -> trajectories
-> hatching (pre-registered bench, LoRA, paired test, promote or roll back).

  python demo/run_demo.py --backend fake                  # CPU, seconds, used by CI
  python demo/run_demo.py --backend hf --model meta-models/Muse-Glimmer-30B   # real run, 1 GPU (80 GB)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory import MemoryStore, TraceIds, get_embedder, TenantIsolationError  # noqa: E402
from orchestrator import Orchestrator, TrajectoryStore, FakeWorker  # noqa: E402
from hatch.bench import preregister  # noqa: E402
from hatch.hatch import Hatcher, bottleneck  # noqa: E402
from demo.tenants import TENANTS, make_tasks, check  # noqa: E402

PROJECT = "hatch-demo"


class Logger:
    def __init__(self, path: Path):
        self.f = path.open("a")

    COLORS = (("PROMOTE", "32;1"), ("ROLLBACK", "31;1"), ("blocked", "33;1"), ("| GPU", "36"),
              ("level 1", "35"), ("journal[", "34;1"), ("pre-registered", "33"))

    def __call__(self, msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        shown = f"{time.strftime('%H:%M:%S')} {msg}"
        for key, code in self.COLORS:
            if key in msg:
                shown = f"\033[{code}m{shown}\033[0m"
                break
        print(shown, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


class FakeTrainer:
    """Stands in for LoRA training on CPU: a real adapter raises the fake worker's accuracy,
    a negative-control adapter lowers it."""

    def __init__(self, worker: FakeWorker):
        self.worker = worker

    def train(self, examples, name, out_dir, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter_config.json").write_text(json.dumps({"fake": True, "n": len(examples)}))
        control = all(e["from"] == "shuffled" for e in examples)
        self.worker.adapter_accuracy[name] = 0.25 if control else 0.95
        log(f"  [{name}] fake training on {len(examples)} examples")
        return {"steps": 0, "n_examples": len(examples), "fake": True}

    def discard(self, name):
        self.worker.adapter_accuracy.pop(name, None)


class ReplayWorker:
    """Replays the raw outputs that Muse Glimmer 30B produced in the recorded GPU run
    (demo/evidence/tenants_jsonl). Same task ids, same adapters -> same verdicts, on a CPU."""

    def __init__(self, evidence: Path, pace: float = 0.0):
        self.name = "replay:Muse-Glimmer-30B(GPU run 2026-11-04)"
        self.out, self.pace = {}, pace
        for f in (evidence / "tenants_jsonl").glob("*/trajectories__*.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                self.out[(r["task_id"], r["adapter"])] = r["raw_output"]

    def run_batch(self, reqs):
        time.sleep(self.pace)
        return [self.out[(r.task_id, r.adapter)] for r in reqs]


class ReplayTrainer:
    """Prints the real, timestamped training lines of the GPU run for this adapter."""

    def __init__(self, evidence: Path, pace: float = 0.0):
        self.lines = (evidence / "real_run" / "run_log.txt").read_text().splitlines()
        self.journals = {}
        for f in (evidence / "real_run" / "tenants").glob("*/hatch_journal.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["event"] == "train_end":
                    self.journals[r["adapter"]] = r
        self.pace = pace

    def train(self, examples, name, out_dir, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"  LoRA r=16 on Muse Glimmer 30B, {len(examples)} examples "
            f"({sum(e['from'] == 'correction' for e in examples)} tenant corrections) -- replaying the GPU log:")
        for l in self.lines:
            if f"[{name}] step" in l:
                log(f"  | GPU {l}")
                time.sleep(self.pace)
        j = self.journals.get(name, {})
        return {k: j[k] for k in ("steps", "loss_first10", "loss_last10", "train_seconds", "trainable_params") if k in j}

    def discard(self, name):
        pass


def load_hf(model_id: str, log):
    import torch
    from transformers import AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token})
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
            log(f"loaded {model_id} with {cls_name} -> {type(model).__name__}")
            break
        except (ValueError, KeyError) as e:
            log(f"  {cls_name} cannot load {model_id}: {str(e)[:120]}")
    if model is None:
        raise RuntimeError(f"cannot load {model_id}")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    log(f"parameters loaded: {n / 1e9:.2f} B on {model.device}")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fake", "hf", "replay"], default="fake")
    ap.add_argument("--evidence", default=str(ROOT / "demo" / "evidence"))
    ap.add_argument("--pace", type=float, default=0.0, help="seconds of pause per batch / log line (screen recording)")
    ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--out", default=str(ROOT / "runs" / "latest"))
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--n-traj", type=int, default=160)
    ap.add_argument("--n-bench", type=int, default=80)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--round2", type=int, default=0, help="tasks served under the new adapter after hatching")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run_log.txt")
    run_uid = os.environ.get("HATCH_RUN_UID") or uuid.uuid4().hex[:6]
    t_start = time.time()
    log(f"=== nika-os-hatch demo | backend={args.backend} model={args.model if args.backend != 'fake' else 'fake'} "
        f"run_uid={run_uid} python={platform.python_version()}")
    embedder = get_embedder(args.embedder)
    mem_root = out / "tenants"

    # ------------------------------------------------------------- 1. memory
    stores, trajs_db, orchs = {}, {}, {}
    for tid, spec in TENANTS.items():
        ms = MemoryStore(mem_root, tid, embedder)
        if ms.count() == 0:
            ids = TraceIds(PROJECT, "handbook-ingest", f"ingest-{tid}", run_uid)
            for src, text in spec["handbook"]:
                ms.add(text, source=src, ids=ids, kind="rule", meta={"pinned": src.endswith("format.md")})
        stores[tid] = ms
        trajs_db[tid] = TrajectoryStore(mem_root, tid)
        orchs[tid] = Orchestrator(ms, trajs_db[tid], spec["display"],
                                  lambda e, a, _t=tid: check(_t, e, a), k=args.k, batch_size=args.gen_batch)
        log(f"[{tid}] memory: {ms.count()} memories in {ms.path.relative_to(out)} (embedder {embedder.name})")

    # isolation probe: the same query in both tenants only ever returns the tenant's own rows
    iso = {}
    for tid, ms in stores.items():
        for q in ("spindle noise work-order code", "days for an AI training"):
            hits = ms.search(q, k=3)
            iso.setdefault(tid, {})[q] = [{"source": h.source, "tenant": h.tenant_id} for h in hits]
            assert all(h.tenant_id == tid for h in hits)
    log("isolation probe: every hit of every query belongs to the querying tenant")
    q = "spindle noise work-order code"
    hits = stores["consulting-firm"].search(q, k=2)
    log(f"[consulting-firm] searches {q!r} (a machine-shop topic) -> {[h.source + ' @' + h.tenant_id for h in hits]}")

    # ------------------------------------------------------------- 2. collect trajectories
    collect = {tid: make_tasks(tid, args.n_traj, seed=1) for tid in TENANTS}
    bench = {tid: make_tasks(tid, args.n_bench, seed=2, exclude_texts={t.text for t in collect[tid]}, prefix="bench-")
             for tid in TENANTS}
    live = {tid: make_tasks(tid, 1, seed=3, prefix="live-") for tid in TENANTS}

    if args.backend == "replay":
        worker = ReplayWorker(Path(args.evidence), args.pace)
        trainer = ReplayTrainer(Path(args.evidence), args.pace / 3)
        model_id = args.model
    elif args.backend == "fake":
        oracle = {t.task_id: t.expected for d in (collect, bench, live) for ts in d.values() for t in ts}
        worker = FakeWorker(oracle, accuracy=0.5)
        trainer = FakeTrainer(worker)
        model_id = "fake"
    else:
        from orchestrator.hf_worker import HFWorker
        from hatch.train_lora import LoraTrainer

        t0 = time.time()
        model, tok = load_hf(args.model, log)
        log(f"model load: {time.time() - t0:.0f}s")
        worker = HFWorker(model, tok, name=f"hf:{args.model}", max_new_tokens=args.max_new_tokens)
        trainer = LoraTrainer(model, tok, steps=args.steps, batch=args.batch, lr=args.lr)
        model_id = args.model

        class _Sync:  # keep the worker on the (possibly PEFT-wrapped) model and the loaded adapters
            def __init__(self, tr, wk):
                self.tr, self.wk = tr, wk

            def train(self, *a, **k):
                r = self.tr.train(*a, **k)
                self.wk.model, self.wk.adapters = self.tr.model, self.tr.loaded
                return r

            def discard(self, name):
                self.tr.discard(name)
                self.wk.adapters = set(self.tr.loaded)

        trainer = _Sync(trainer, worker)

    for tid in TENANTS:
        ids = TraceIds(PROJECT, "collect", f"collect-{tid}", run_uid)
        t0 = time.time()
        tr = orchs[tid].run(collect[tid], [worker], ids, adapter=None, log=log)
        ok = sum(t.verdict for t in tr)
        log(f"[{tid}] collected {len(tr)} trajectories, pass {ok}/{len(tr)} ({time.time() - t0:.0f}s)")
        ex = tr[0]
        log(f"[{tid}] trajectory #{ex.id}: ids={ex.ids.project_id}/{ex.ids.job_id}/{ex.ids.task_id}/{ex.ids.run_uid}")
        log(f"    request : {collect[tid][0].text}")
        log(f"    tools   : memory.search -> {[h['source'] for h in ex.tool_calls[0]['result']]}")
        log(f"    answer  : {ex.answer}  verdict={'PASS' if ex.verdict else 'FAIL'}")
        from hatch.reflect import propose_pins
        from hatch.registry import HatchJournal
        rp = propose_pins(tr, stores[tid])
        HatchJournal(mem_root, tid).write("reflect_level1", **rp)
        log(f"[{tid}] level 1 (prompt evolution) field fail rates {rp['field_fail_rates']} -> pin "
            f"{[p['pin'][0]['source'] for p in rp['proposals'] if p['pin']]} [{rp['status']}]")

    # ------------------------------------------------------------- 3. hatch
    results = {}
    for tid in TENANTS:
        trs = [t for t in trajs_db[tid].all(job_id="collect")]
        b = bottleneck(trs, min_trajectories=min(50, args.n_traj), max_fail_rate=0.2)
        hatcher_log = log
        hatcher = None

        def evaluate(tasks, adapter, _tid=tid):
            ids = TraceIds(PROJECT, "bench", f"bench-{_tid}-{adapter or 'base'}", run_uid)
            res = orchs[_tid].run(tasks, [worker], ids, adapter=adapter, log=log)
            return {t.task_id: t.verdict for t in res}

        hatcher = Hatcher(mem_root, tid, trainer, evaluate, log=hatcher_log)
        hatcher.journal.write("trigger", **b)
        log(f"[{tid}] bottleneck: fail rate {b['fail_rate']:.3f} on {b['n']} trajectories -> hatch={b['hatch']}")
        results[tid] = {"bottleneck": b, "hatches": []}
        if not b["hatch"]:
            continue
        pre = preregister(out / "tenants" / tid / "bench_prereg.json", tid, bench[tid], model_id,
                          extra={"train_trajectories": len(trs), "lora_steps": args.steps})
        hatcher.journal.write("preregistered", **pre)
        log(f"[{tid}] bench pre-registered: {pre['n_items']} items, sha256 {pre['sha256'][:16]}")
        plan = ([("control", True)] if (tid == "machine-shop" and not args.no_control) else []) + [("v1", False)]
        for suffix, control in plan:
            name = f"{tid}-{suffix}"
            r = hatcher.hatch(trs, bench[tid], out / "tenants" / tid / "bench_prereg.json", name, control=control,
                              train_kwargs={"base_model": model_id})
            results[tid]["hatches"].append(r)
        results[tid]["active_adapter"] = hatcher.registry.active

    # ------------------------------------------------------------- 4. round 2: work continues under the new adapter
    if args.round2 > 0:
        from hatch.registry import AdapterRegistry, HatchJournal
        for tid in TENANTS:
            active = AdapterRegistry(mem_root, tid).active
            tasks2 = make_tasks(tid, args.round2, seed=4, prefix="r2-",
                                exclude_texts={t.text for t in collect[tid] + bench[tid]})
            if args.backend == "fake":
                worker.oracle.update({t.task_id: t.expected for t in tasks2})
            ids = TraceIds(PROJECT, "collect-r2", f"collect-r2-{tid}", run_uid)
            t0 = time.time()
            tr2 = orchs[tid].run(tasks2, [worker], ids, adapter=active, log=log)
            b2 = bottleneck(tr2, min_trajectories=min(50, args.round2), max_fail_rate=0.2)
            HatchJournal(mem_root, tid).write("trigger", round=2, adapter=active, **b2)
            lat = sum(t.latency_s for t in tr2) / max(1, len(tr2))
            log(f"[{tid}] round 2 under adapter={active}: pass {sum(t.verdict for t in tr2)}/{len(tr2)}, "
                f"fail rate {b2['fail_rate']:.3f} -> hatch={b2['hatch']} ({lat:.2f}s/task, {time.time() - t0:.0f}s)")
            results[tid]["round2"] = {"adapter": active, "bottleneck": b2, "latency_s_per_task": lat}

    # ------------------------------------------------------------- 5. serve with per-tenant adapters
    served = {}
    for tid in TENANTS:
        from hatch.registry import AdapterRegistry
        active = AdapterRegistry(mem_root, tid).active
        t = live[tid][0]
        tr = orchs[tid].run([t], [worker], TraceIds(PROJECT, "serve", f"serve-{tid}", run_uid), adapter=active)[0]
        served[tid] = {"adapter": active, "task": t.text, "answer": tr.answer, "verdict": tr.verdict}
        log(f"[{tid}] served with adapter={active}: {t.text!r} -> {tr.answer} ({'pass' if tr.verdict else 'fail'})")

    # cross-tenant attempt: opening tenant B's file as tenant A must fail
    import shutil
    try:
        shutil.copy(stores["consulting-firm"].path, out / "stolen.sqlite")
        (mem_root / "intruder").mkdir(exist_ok=True)
        shutil.copy(out / "stolen.sqlite", mem_root / "intruder" / "memory.sqlite")
        MemoryStore(mem_root, "intruder", embedder)
        cross = "NOT BLOCKED"
    except TenantIsolationError as e:
        cross = f"blocked: {e}"
    finally:
        (out / "stolen.sqlite").unlink(missing_ok=True)
        shutil.rmtree(mem_root / "intruder", ignore_errors=True)
    log(f"consulting-firm memory file copied under another tenant id -> {cross}")

    for tid in TENANTS:
        from hatch.registry import HatchJournal
        for e in HatchJournal(mem_root, tid).read():
            if e["event"] in ("preregistered", "promote", "rollback"):
                extra = (f"sha256={e['sha256'][:12]}" if e["event"] == "preregistered" else
                         f"{e['adapter']} {e['base_acc']:.2f}->{e['cand_acc']:.2f} p={e['p_sign']:.2g}")
                log(f"journal[{tid}] {e['event']:<13} {extra}")
    summary = {"run_uid": run_uid, "backend": args.backend, "model": model_id, "args": vars(args),
               "duration_s": round(time.time() - t_start, 1), "isolation_probe": iso, "cross_tenant_file": cross,
               "results": results, "served": served}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    log(f"=== done in {summary['duration_s']}s -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
