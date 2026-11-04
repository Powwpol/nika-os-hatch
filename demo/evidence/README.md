# Evidence of the real run (4 Nov 2026)

Model `meta-models/Muse-Glimmer-30B`, bf16, one RTX PRO 6000 (96 GB). Command: see the main README.

| Path | Content |
|---|---|
| `real_run/run_log.txt` | timestamped log of the whole run (collection, training steps, verdicts, serving, isolation checks) |
| `real_run/stdout_stderr.txt` | raw console output (library warnings included, progress bars removed) |
| `real_run/summary.json` | arguments, results of every hatch, served answers, isolation checks |
| `real_run/tenants/<tenant>/bench_prereg.json` | the pre-registered protocol and its sha256, written before training |
| `real_run/tenants/<tenant>/hatch_journal.jsonl` | trigger, train start/end, promote or rollback, with full statistics |
| `real_run/tenants/<tenant>/adapters/` | registry (active adapter, history) and adapter configs (weights not committed) |
| `tenants_jsonl/<tenant>/` | every memory and every trajectory (prompt, tool calls, raw output, answer, verdict, ids), per job and adapter |
| `trajectory_stats.json` | per job and adapter: pass rate, per-field accuracy, mean latency |
| `0*.png` | charts rendered by `demo/render_evidence.py` |

The SQLite stores themselves are not committed; the JSONL files are complete exports of them (embeddings left out).
