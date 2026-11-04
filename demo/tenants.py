"""Two synthetic tenants with their own private conventions.

Everything here is invented for the demo: no real company, person or record.
Each tenant has
  * a handbook (the conventions a new employee would have to learn), stored in
    that tenant's memory and retrieved at run time;
  * a task generator: free-text requests -> the structured answer the tenant expects;
  * a deterministic checker (field-by-field exact match).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Task:
    tenant_id: str
    task_id: str
    text: str
    expected: dict
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- machine shop
MACHINE_SHOP_HANDBOOK = [
    ("handbook/codes.md#spindle", "Work-order code family SPN = spindle problems (noise, vibration, runout, bearing heat)."),
    ("handbook/codes.md#coolant", "Work-order code family CLT = coolant problems (leak, low level, foam, smell, pump)."),
    ("handbook/codes.md#hydraulic", "Work-order code family HYD = hydraulic problems (pressure drop, oil leak, slow clamp, hose)."),
    ("handbook/codes.md#electrical", "Work-order code family ELE = electrical problems (breaker trip, drive fault, cabinet alarm, sensor cable)."),
    ("handbook/codes.md#tooling", "Work-order code family TOL = tooling problems (broken insert, tool wear, wrong offset, holder damage)."),
    ("handbook/codes.md#guards", "Work-order code family SAF = safety guard or door interlock problems. SAF always wins over any other family."),
    ("handbook/codes.md#suffix", "The work-order code is FAMILY-SHIFT, where SHIFT is 1 for the morning shift, 2 for the afternoon shift and 3 for the night shift. Example: CLT-2."),
    ("handbook/priority.md", "Priority P1 if the machine is stopped or if it is a safety (SAF) issue; P2 if the machine still runs but quality or noise is degraded; P3 for anything that can wait for planned maintenance."),
    ("handbook/assets.md#cnc", "Asset ids: machining centres are CNC-01 to CNC-06 (operators say 'CNC 4', 'mill #4', 'the fourth machining centre')."),
    ("handbook/assets.md#lathe", "Asset ids: lathes are LTH-01 to LTH-03 (operators say 'lathe 2', 'turning centre 2')."),
    ("handbook/assets.md#press", "Asset ids: presses are PRS-01 and PRS-02 (operators say 'press 1', 'the big press' = PRS-02)."),
    ("handbook/format.md", 'Answer format: {"code": "FAM-S", "priority": "P1|P2|P3", "asset": "XXX-NN"} and nothing else.'),
]

_MS_SYMPTOMS = {
    "SPN": ["spindle is making a grinding noise", "spindle vibrates above 8000 rpm", "spindle bearing feels very hot",
            "runout on the spindle is out of tolerance"],
    "CLT": ["coolant is leaking under the chip conveyor", "coolant level alarm keeps coming back",
            "coolant smells bad and foams", "coolant pump makes a whining sound"],
    "HYD": ["hydraulic pressure drops during clamping", "oil leak at the hydraulic unit",
            "clamping is very slow this morning", "a hydraulic hose is sweating oil"],
    "ELE": ["main breaker tripped twice", "axis drive fault on X", "electrical cabinet over-temperature alarm",
            "probe sensor cable is damaged"],
    "TOL": ["insert broke on the roughing tool", "tool wear is much faster than usual", "wrong tool offset loaded",
            "tool holder is chipped"],
    "SAF": ["door interlock does not lock", "safety guard is cracked", "light curtain is bypassed", "the guard switch is loose"],
}
_MS_STATE = {
    "P1": ["machine is down", "the machine stopped", "production is stopped on it", "it will not restart"],
    "P2": ["still running but parts are rough", "running, but noisy", "still producing with degraded finish",
           "it runs but scrap rate is up"],
    "P3": ["no impact for now, can wait", "it can wait for the planned stop", "minor, fix at next maintenance",
           "no urgency"],
}
_MS_SHIFT = {1: ["Morning shift", "6am team", "Morning crew"], 2: ["Afternoon shift", "2pm team", "Afternoon crew"],
             3: ["Night shift", "10pm team", "Night crew"]}
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def _ms_asset(rng):
    kind = rng.choice(["CNC", "CNC", "LTH", "PRS"])
    if kind == "CNC":
        n = rng.randint(1, 6)
        say = rng.choice([f"CNC {n}", f"mill #{n}", f"the {_ORD[n]} machining centre"])
    elif kind == "LTH":
        n = rng.randint(1, 3)
        say = rng.choice([f"lathe {n}", f"turning centre {n}"])
    else:
        n = rng.randint(1, 2)
        say = "the big press" if (n == 2 and rng.random() < 0.5) else f"press {n}"
    return f"{kind}-{n:02d}", say


def machine_shop_task(rng: random.Random, i: int) -> Task:
    fam = rng.choice(list(_MS_SYMPTOMS))
    pri = rng.choice(["P1", "P2", "P3"])
    shift = rng.randint(1, 3)
    asset, say = _ms_asset(rng)
    state = rng.choice(_MS_STATE[pri])
    if fam == "SAF":
        pri = "P1"
    text = f"{rng.choice(_MS_SHIFT[shift])}: {say}, {rng.choice(_MS_SYMPTOMS[fam])}; {state}."
    return Task("machine-shop", f"ms-{i:04d}", text, {"code": f"{fam}-{shift}", "priority": pri, "asset": asset})


# --------------------------------------------------------------------------- consulting firm
CONSULTING_HANDBOOK = [
    ("handbook/catalog.md#diag", "Service DIAG = data and AI diagnostic of a client company (audit of data, processes and quick wins)."),
    ("handbook/catalog.md#train", "Service TRAIN = AI training for client staff (workshops for employees or managers)."),
    ("handbook/catalog.md#lean", "Service LEAN = process mapping and value-stream mapping of a workshop or office flow."),
    ("handbook/catalog.md#proto", "Service PROTO = a working prototype (agent, dashboard, automation) built for the client."),
    ("handbook/catalog.md#bid", "Service BID = help to answer a public tender or call for proposals."),
    ("handbook/deliverables.md", "Default deliverable: DIAG -> report, TRAIN -> workshop, LEAN -> map, PROTO -> prototype, BID -> proposal."),
    ("handbook/days.md#diag", "Days for DIAG: 3 days, plus 1 day if the client has more than 50 employees."),
    ("handbook/days.md#train", "Days for TRAIN: 1 day per group of up to 10 attendees (e.g. 25 attendees = 3 days)."),
    ("handbook/days.md#lean", "Days for LEAN: 2 days per site to map."),
    ("handbook/days.md#proto", "Days for PROTO: 5 days, or 8 days if the prototype must connect to the client's ERP."),
    ("handbook/days.md#bid", "Days for BID: 2 days whatever the tender."),
    ("handbook/format.md", 'Answer format: {"service": "DIAG|TRAIN|LEAN|PROTO|BID", "deliverable": "...", "days": N} and nothing else.'),
]
_CF_DELIV = {"DIAG": "report", "TRAIN": "workshop", "LEAN": "map", "PROTO": "prototype", "BID": "proposal"}
_CF_CLIENT = ["a packaging plant", "a regional bakery chain", "a textile mill", "an engineering school",
              "a logistics SME", "a furniture maker", "a plastics injector", "a water utility"]


def consulting_task(rng: random.Random, i: int) -> Task:
    svc = rng.choice(list(_CF_DELIV))
    client = rng.choice(_CF_CLIENT)
    if svc == "DIAG":
        emp = rng.choice([12, 30, 45, 50, 60, 80, 140, 300])
        text = rng.choice([f"{client} with {emp} employees wants to know where AI and its data could help first.",
                           f"Can we audit the data and processes of {client}? They have {emp} people.",
                           f"{client} ({emp} staff) asks for an assessment of their data before investing in AI."])
        days = 3 + (1 if emp > 50 else 0)
    elif svc == "TRAIN":
        n = rng.choice([6, 10, 12, 18, 20, 25, 31, 40])
        text = rng.choice([f"{client} wants to train {n} employees to use generative AI safely.",
                           f"Please plan AI sessions for {n} managers at {client}.",
                           f"{n} people at {client} need hands-on training on AI assistants."])
        days = -(-n // 10)
    elif svc == "LEAN":
        sites = rng.choice([1, 2, 3, 4])
        text = rng.choice([f"{client} wants to map the flow of orders across its {sites} site(s).",
                           f"Value-stream mapping needed for {client}, {sites} site(s) involved.",
                           f"{client} asks us to draw how work moves through its {sites} site(s)."])
        days = 2 * sites
    elif svc == "PROTO":
        erp = rng.random() < 0.5
        text = rng.choice([f"{client} wants a working agent that drafts quotes" + (", plugged into their ERP." if erp else ", standalone for now."),
                           f"Build a small dashboard prototype for {client}" + (" that reads from their ERP." if erp else " from spreadsheets."),
                           f"{client} asks for an automation prototype" + (" connected to the ERP." if erp else " without any ERP link.")])
        days = 8 if erp else 5
    else:
        text = rng.choice([f"{client} must answer a public tender next month and wants our help.",
                           f"Help {client} reply to a call for proposals from the region.",
                           f"{client} found a public procurement notice and needs a proposal written."])
        days = 2
    return Task("consulting-firm", f"cf-{i:04d}", text, {"service": svc, "deliverable": _CF_DELIV[svc], "days": days})


TENANTS = {
    "machine-shop": {"display": "machine shop maintenance desk", "handbook": MACHINE_SHOP_HANDBOOK, "gen": machine_shop_task},
    "consulting-firm": {"display": "consulting firm intake desk", "handbook": CONSULTING_HANDBOOK, "gen": consulting_task},
}


def make_tasks(tenant_id: str, n: int, seed: int, exclude_texts: set[str] | None = None, prefix: str = "") -> list[Task]:
    rng = random.Random(f"{tenant_id}:{seed}")
    gen = TENANTS[tenant_id]["gen"]
    out, seen, i = [], set(exclude_texts or ()), 0
    while len(out) < n:
        t = gen(rng, i)
        i += 1
        if t.text in seen:
            continue
        seen.add(t.text)
        t.task_id = f"{prefix}{t.task_id}"
        out.append(t)
    return out


def check(tenant_id: str, expected: dict, answer: dict | None) -> tuple[bool, dict]:
    """Deterministic verifier: every expected field must match exactly."""
    if not isinstance(answer, dict):
        return False, {"reason": "no parsable JSON answer"}
    fields = {k: (str(answer.get(k)).strip().upper() == str(v).strip().upper()) for k, v in expected.items()}
    return all(fields.values()), {"fields": fields}


def gold_answer(task: Task) -> str:
    return json.dumps(task.expected, ensure_ascii=False)
