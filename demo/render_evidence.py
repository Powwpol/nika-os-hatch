#!/usr/bin/env python3
"""Render evidence images (1920x1080) from a finished run directory.

  python demo/render_evidence.py runs/latest demo/evidence
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BG, FG, MUTED, A, B, OK, KO = "#0f1419", "#e6e1cf", "#8a9199", "#7DB5A5", "#E99971", "#9ccc65", "#ef6b73"
W, H, DPI = 19.2, 10.8, 100


def fig():
    f = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    return f


def title(f, text, sub=None):
    f.text(0.04, 0.93, text, color=FG, fontsize=34, weight="bold", family="DejaVu Sans")
    if sub:
        f.text(0.04, 0.885, sub, color=MUTED, fontsize=17, family="DejaVu Sans")
    f.text(0.96, 0.03, "nika-os-hatch · built during the Meta Global AI Developer Hackathon", color=MUTED,
           fontsize=12, ha="right")


def bench_chart(summary, out):
    rows = []
    for tid, r in summary["results"].items():
        for h in r["hatches"]:
            rows.append((tid, h))
    f = fig()
    title(f, "Hatch verdicts on the pre-registered paired bench",
          f"model: {summary['model']} · bench items per tenant: {summary['args']['n_bench']} · "
          f"exact sign test + paired bootstrap, alpha 0.05")
    ax = f.add_axes([0.06, 0.12, 0.6, 0.68], facecolor=BG)
    labels, base, cand, cols = [], [], [], []
    for tid, h in rows:
        v = h["verdict"]
        labels.append(f"{tid}\n{h['adapter'].split('-')[-1]}{' (negative control)' if h['control'] else ''}")
        base.append(v["base_acc"]), cand.append(v["cand_acc"])
        cols.append(OK if h["decision"] == "promote" else KO)
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], base, 0.38, color=MUTED, label="before (base / active adapter)")
    ax.bar([i + 0.2 for i in x], cand, 0.38, color=cols, label="candidate adapter")
    for i, (b, c) in enumerate(zip(base, cand)):
        ax.text(i - 0.2, b + 0.02, f"{b:.2f}", color=FG, ha="center", fontsize=15)
        ax.text(i + 0.2, c + 0.02, f"{c:.2f}", color=FG, ha="center", fontsize=15, weight="bold")
    ax.set_xticks(list(x), labels, color=FG, fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("exact-match accuracy", color=FG, fontsize=15)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=13, loc="upper left")
    y = 0.78
    for tid, h in rows:
        v = h["verdict"]
        col = OK if h["decision"] == "promote" else KO
        f.text(0.69, y, f"{h['adapter']}", color=FG, fontsize=17, weight="bold")
        f.text(0.69, y - 0.035, f"{h['decision'].upper()}", color=col, fontsize=19, weight="bold")
        f.text(0.69, y - 0.07, f"wins {v['wins']} · losses {v['losses']} · ties {v['ties']}", color=FG, fontsize=14)
        f.text(0.69, y - 0.10, f"sign test p = {v['p_sign']:.2g} · bootstrap lower = {v['boot_lower']:+.3f}",
               color=FG, fontsize=14)
        f.text(0.69, y - 0.13, f"prereg sha256 {v['prereg_sha256'][:16]}…", color=MUTED, fontsize=12, family="monospace")
        y -= 0.21
    f.savefig(out, facecolor=BG)
    plt.close(f)


def loss_chart(run_log, out, summary):
    curves = {}
    for line in run_log.splitlines():
        m = re.search(r"\[([\w\-]+)\] step (\d+)/(\d+) loss ([\d.]+)", line)
        if m:
            curves.setdefault(m.group(1), []).append((int(m.group(2)), float(m.group(4))))
    f = fig()
    title(f, "LoRA training on each tenant's own trajectories",
          f"{summary['model']} · bf16 · LoRA r=16 on attention projections of the language model")
    ax = f.add_axes([0.07, 0.12, 0.88, 0.68], facecolor=BG)
    pal = [A, B, "#c3a6ff", OK]
    for i, (name, pts) in enumerate(curves.items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=pal[i % len(pal)], lw=3, marker="o", label=name)
    ax.set_xlabel("step", color=FG, fontsize=15)
    ax.set_ylabel("loss on answer tokens", color=FG, fontsize=15)
    ax.tick_params(colors=FG, labelsize=13)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=14)
    f.savefig(out, facecolor=BG)
    plt.close(f)


def terminal(lines, out, head):
    f = fig()
    title(f, head)
    y = 0.84
    for line in lines[-34:]:
        col = OK if "PROMOTE" in line else KO if "ROLLBACK" in line else FG
        f.text(0.04, y, line[:150], color=col, fontsize=12.5, family="DejaVu Sans Mono")
        y -= 0.023
    f.savefig(out, facecolor=BG)
    plt.close(f)


def trajectory_card(run, tid, out):
    db = sqlite3.connect(Path(run) / "tenants" / tid / "trajectories.sqlite")
    r = db.execute("SELECT task_id, project_id, job_id, run_task_id, run_uid, worker, adapter, messages, tool_calls,"
                   " raw_output, answer, verdict, expected FROM trajectory WHERE job_id='collect' ORDER BY id LIMIT 1").fetchone()
    msgs = json.loads(r[7])
    tools = json.loads(r[8])
    f = fig()
    title(f, f"One recorded trajectory · tenant {tid}", "prompt, tool calls, raw output, verdict and four trace ids")
    txt = [f"ids     project={r[1]}  job={r[2]}  task={r[3]}  run={r[4]}",
           f"worker  {r[5]}   adapter={r[6]}",
           "", "user    " + msgs[1]["content"], "", "memory.search ->"]
    txt += [f"   #{h['memory_id']} {h['source']} (score {h['score']})" for h in tools[0]["result"]]
    raw = r[9].replace("\n", " ")
    txt += ["", "raw output (truncated):"] + textwrap.wrap(raw[:600], 140)
    txt += ["", f"answer   {r[10]}", f"expected {r[12]}", f"verdict  {'PASS' if r[11] else 'FAIL'}"]
    y = 0.84
    for line in txt:
        f.text(0.04, y, line, color=FG if "verdict" not in line else (OK if r[11] else KO), fontsize=13.5,
               family="DejaVu Sans Mono")
        y -= 0.027
    f.savefig(out, facecolor=BG)
    plt.close(f)


def architecture(out):
    f = fig()
    title(f, "Architecture", "everything in this diagram was written during the hackathon (this repository)")
    ax = f.add_axes([0, 0, 1, 0.85])
    ax.set_xlim(0, 100), ax.set_ylim(0, 60), ax.axis("off")

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=BG, ec=c, lw=3))
        ax.text(x + w / 2, y + h / 2, t, color=FG, ha="center", va="center", fontsize=15, wrap=True)

    def arrow(x1, y1, x2, y2, t=""):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=2.5))
        if t:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, t, color=MUTED, ha="center", fontsize=12)

    for i, (tid, c) in enumerate((("tenant: machine-shop", A), ("tenant: consulting-firm", B))):
        y0 = 36 - i * 26
        ax.text(2, y0 + 19, tid, color=c, fontsize=17, weight="bold")
        box(2, y0, 16, 14, "memory/\nSQLite + embeddings\nsource + 4 trace ids", c)
        box(24, y0, 18, 14, "orchestrator/\nplan · memory.search\ndispatch · verify", c)
        box(48, y0, 16, 14, "trajectories\nprompt · tools\noutput · verdict", c)
        box(70, y0, 26, 14, "hatch/\ntrigger · pre-registered bench\nLoRA · paired test\npromote or roll back", c)
        arrow(18.6, y0 + 7, 23.4, y0 + 7, "retrieve")
        arrow(42.6, y0 + 7, 47.4, y0 + 7, "record")
        arrow(64.6, y0 + 7, 69.4, y0 + 7, "train")
    ax.text(50, 27.5, "headless worker: Muse Glimmer 30B (Meta open weights) · one LoRA adapter per tenant, "
            "served only inside its tenant", color=FG, ha="center", fontsize=15, style="italic")
    f.savefig(out, facecolor=BG)
    plt.close(f)


def main(run, outdir):
    run, outdir = Path(run), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    log = (run / "run_log.txt").read_text()
    architecture(outdir / "01_architecture.png")
    trajectory_card(run, "machine-shop", outdir / "02_trajectory_machine_shop.png")
    loss_chart(log, outdir / "03_lora_loss.png", summary)
    bench_chart(summary, outdir / "04_bench_verdicts.png")
    key = [l for l in log.splitlines() if "batch" not in l and "step " not in l]
    terminal(key, outdir / "05_run_log.png", "Run log (real run, excerpt)")
    print("rendered", sorted(p.name for p in outdir.glob("*.png")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
