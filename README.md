# nika-os-hatch

**Persistent memory per tenant, an orchestrator for headless workers, and a *hatching* loop that fine-tunes a Meta open-weight model on each tenant's own work, then keeps the result only if a pre-registered paired test proves it is better.**

Built during the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai) (Oct 27 – Nov 4, 2026) by BCUB3 (Paul Obara).
Base model: **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`, Meta open weights, Apache-2.0), run in bf16 on one rented GPU.

## What is new

**Everything in this repository was written during the hackathon.** It is a small, self-contained, open reference implementation of a pattern we run in production at BCUB3 (our internal operating layer, *Nika OS*). None of Nika OS's code is in this repository and none of it is needed to run it. The tenants and the data are synthetic.

## The idea in one paragraph

General assistants forget, and they never get better at *your* job. Here every tenant (a user, a team, a company) gets its own memory and its own trajectory log. An orchestrator runs the tenant's tasks on headless workers and records each run (prompt, tool calls, output, verdict, trace ids). When a bottleneck is *measured* (too many failed trajectories), the tenant's agent **hatches**: a benchmark is pre-registered and hashed, a LoRA adapter is trained on the tenant's own trajectories, and the adapter is promoted only if a paired test without any normality assumption says it wins. Otherwise it is rolled back and the tenant keeps what it had. Adapters, like memories, never leave their tenant.

## Result of the real run (Muse Glimmer 30B, 1 GPU)

Run of 4 Nov 2026, one RTX PRO 6000 (96 GB), bf16, 20.5 minutes end to end. 240 trajectories collected per tenant with the base model, then a bench of 100 held-out items per tenant, pre-registered before any training. LoRA r=16, 200 steps, batch 4.

| Tenant · candidate | Before | After | Wins / losses / ties | Sign test p | Bootstrap lower bound of Δ | Decision |
|---|---|---|---|---|---|---|
| machine-shop · **negative control** (shuffled answers) | 0.57 | 0.02 | 1 / 56 / 43 | 1 | −0.630 | **rolled back** |
| machine-shop · v1 | 0.57 | **1.00** | 43 / 0 / 57 | 1.1e-13 | +0.350 | **promoted** |
| consulting-firm · v1 | 0.05 | **0.95** | 90 / 0 / 10 | 8.1e-28 | +0.850 | **promoted** |

Where the gain comes from (bench, per field): the base model already finds the consulting service (0.95) and mostly the day count (0.85) but almost never the firm's own deliverable names (0.05 → 1.00 with the adapter); in the machine shop it misses the work-order code convention (0.65 → 1.00). The specialised adapters also answer without a long reasoning preamble: mean worker time per task in a batch of 48 drops from about 1.1 s to 0.1 s.

Then each tenant is served with its own adapter (`served` in `summary.json`), and a store file copied from one tenant into another is refused (`cross_tenant_file`).
GPU cost of the hackathon runs (two smoke runs + this run): about 2.2 USD.

Evidence (logs, journals, pre-registrations, trajectory databases, charts): [`demo/evidence/`](demo/evidence/).

## Architecture

```
tenant ─┬─ memory/          SQLite file per tenant, local embeddings, every memory has a source + 4 trace ids
        ├─ orchestrator/    plan (memory.search) → dispatch to headless workers → verify → record trajectory
        │                   worker = Muse Glimmer 30B (transformers), one LoRA adapter per tenant
        └─ hatch/           trigger → pre-registered bench (sha256) → LoRA (PEFT) → paired test → promote | roll back
```

| Folder | What it does |
|---|---|
| `memory/` | `MemoryStore` is bound to one tenant when opened; reads take no tenant argument; every row re-checks its tenant; a store file copied into another tenant is refused; tenant ids cannot escape their directory. Writes without `source` or any of the four ids (`project_id`, `job_id`, `task_id`, `run_uid`) are refused. Embedders: feature hashing (default, no download) or any local sentence-transformers model. |
| `orchestrator/` | `Orchestrator` plans each task (pinned memories + top-k retrieval, recorded as a `memory.search` tool call), batches requests to headless workers in parallel, parses the answer from the model's user channel, runs the tenant's checker (`answer.validate`), and stores one trajectory per task. `HFWorker` serves the Meta model with PEFT multi-adapter: the base model or exactly the tenant's active adapter. |
| `hatch/` | `bench.py`: pre-registration (items, metric, test, alpha, seed, decision rule) hashed before training, re-verified before deciding; exact one-sided sign test on discordant pairs + paired bootstrap. `train_lora.py`: LoRA r=16 on the language model's attention projections, loss on answer tokens only. `hatch.py`: trigger, negative control, promote or roll back, refuses bench items that leaked into training. `registry.py`: per-tenant adapter registry and journal. |
| `demo/` | Two synthetic tenants (a machine-shop maintenance desk with its own work-order codes, a consulting firm with its own service catalogue and day rules), `run_demo.py`, `render_evidence.py`. |
| `tests/` | Tenant isolation, trajectory recording, promotion / rollback, tampered pre-registration, bench leakage, end-to-end demo on CPU. |

### Decision rule (pre-registered)

Promote iff Δ accuracy > 0, **and** the exact one-sided sign test on discordant pairs gives p < 0.05, **and** the one-sided 95 % paired-bootstrap lower bound of Δ is > 0. Otherwise roll back. A **negative control** (adapter trained on shuffled answers) runs through the same gate and must be rolled back.

## Run it

CPU, seconds, no model download (fake worker, used by CI):

```bash
pip install -r requirements.txt
pytest -q
python demo/run_demo.py --backend fake --out runs/fake --round2 60
```

Real run with the Meta model (one GPU with ≥ 80 GB, bf16; exactly the command of the run above):

```bash
pip install -r requirements-gpu.txt
hf download meta-models/Muse-Glimmer-30B
python demo/run_demo.py --backend hf --model meta-models/Muse-Glimmer-30B --out runs/real \
  --n-traj 240 --n-bench 100 --steps 200 --batch 4 --gen-batch 48 --max-new-tokens 512
python demo/render_evidence.py runs/real demo/evidence
```

`--round2 N` additionally serves N new tasks per tenant under its promoted adapter and re-runs the bottleneck trigger (tested on CPU; not part of the GPU run above).

## Where the Meta model is used

* `orchestrator/hf_worker.py`: Muse Glimmer 30B is the headless worker for every task, with its own chat template (reasoning channel, then user channel; `Reasoning strength: low`).
* `hatch/train_lora.py`: the LoRA adapters are trained on Muse Glimmer 30B (`model.language_model.*.{q,k,v,o}_proj`).
* `demo/run_demo.py`: loads the model once, collects trajectories, hatches each tenant, then serves each tenant with its own adapter.

We first looked at Muse Spark Open; at 816 B parameters (MoE, A42B) it does not fit a LoRA run on one GPU within our budget, so we used Muse Glimmer 30B, the other open-weight model listed for the hackathon.

## Honest limits

* Synthetic tenants: the bench items are held-out requests from the same generator as the training trajectories. The test measures specialisation to a tenant's conventions, not general capability.
* The base model is not handicapped: it gets the same prompt, the same retrieved memories and the pinned answer format. Its main failures are the tenant's private vocabulary (e.g. the consulting firm's deliverable names) that retrieval did not surface, and long reasoning.
* Retrieval in the real run used the feature-hashing embedder (no neural embedder), on purpose: it keeps the demo free of extra downloads, and its misses are part of what the adapter learns.
* "RL" here is the simplest verifier-driven form (failed trajectories are corrected by the tenant and kept, then supervised fine-tuning); a policy-gradient step is future work.

## Related work by the author

Two French patent applications filed with INPI by Paul Obara / BCUB3 (FR2500517, FR2513029) concern statistical control of how much of a winning variant is merged, and an activation function for small networks. **They are not implemented in this repository.** If you build on those methods, please cite them.

## License

MIT for this repository. Muse Glimmer 30B is distributed by Meta under Apache-2.0; its weights are not included here.
