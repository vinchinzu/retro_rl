#!/usr/bin/env python3
"""Generate disjoint dual-track room-segment task cards from ROOM_WORK_QUEUE.

Each card owns exactly one problem's policy + state + optional note — safe for
N-wide parallel OpenCode dispatch without continuous tip work.

Usage (from repo root):
  uv run python super_metroid/scripts/generate_room_segment_cards.py --count 8
  uv run python super_metroid/scripts/generate_room_segment_cards.py --count 16 --print-ids
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SM = ROOT / "super_metroid"
QUEUE_CSV = SM / "docs" / "routes" / "ROOM_WORK_QUEUE.csv"
TASKS_DIR = SM / "docs" / "tasks"
STATE_DIR = SM / "custom_integrations" / "SuperMetroid-Snes"
POLICY_DIR = SM / "policies" / "room_clears"

# Prefer work that is still open on the practice board.
SKIP_STATUS = {"ready"}

# Explicitly parked residual rooms (Wave 9) — do not re-burn farm rounds.
PARKED_PROBLEMS = {
    "room_a1ad_from_9f64_to_a1d8",  # Boulder — parked
    "room_d21c_from_d3b6_to_d08a",  # Crab Hole — parked
    "room_a865_from_a815_to_a8b9",  # Ice tutorial — parked
    "room_abd2_from_ab64_to_ac00",  # Grapple tut 2 — parked
    "room_b62b_from_b482_to_b5d5",  # Metal Pirates — combat parked
    "room_a890_from_a8b9_to_a8b9",  # Ice Beam collect — parked residual
    "room_a447_from_a408_to_a408",  # Spazer — parked residual
}


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").upper()
    return s[:28] if s else "ROOM"


def _state_exists(problem_id: str, state_file: str) -> bool:
    p = SM / state_file
    if p.is_file():
        return True
    # Common bootstrap name: room_<room>_from_<src>.state
    stem = problem_id.rsplit("_to_", 1)[0]
    return any(STATE_DIR.glob(f"{stem}*.state"))


def _policy_exists(problem_id: str, policy_file: str) -> bool:
    p = SM / policy_file
    if p.is_file():
        return True
    return (POLICY_DIR / f"{problem_id}.json").is_file()


def _policy_verified(problem_id: str, policy_file: str) -> bool:
    """True if practice policy already promoted / verified (skip)."""
    candidates = [SM / policy_file, POLICY_DIR / f"{problem_id}.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # room clear policies use "verified" or similar status fields
        if '"verified_development_state"' in text or '"verified": true' in text:
            return True
        if re.search(r'"status"\s*:\s*"verified', text):
            return True
    return False


def _load_queue() -> list[dict[str, str]]:
    with QUEUE_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _existing_card_problem_ids() -> set[str]:
    """Problem ids already claimed by SM-ROOM-SEG-*.md cards (open or done)."""
    claimed: set[str] = set()
    for path in TASKS_DIR.glob("SM-ROOM-SEG-*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"`(room_[a-z0-9_]+)`", text)
        if m:
            claimed.add(m.group(1))
        m2 = re.search(r"problem `?(room_[a-z0-9_]+)`?", text)
        if m2:
            claimed.add(m2.group(1))
    return claimed


def _next_card_index() -> int:
    best = 0
    for path in TASKS_DIR.glob("SM-ROOM-SEG-*.md"):
        m = re.search(r"SM-ROOM-SEG-(\d+)", path.stem)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def _select_rows(
    rows: list[dict[str, str]],
    *,
    count: int,
    queues: set[str],
    claimed: set[str],
    prefer_teleport: bool,
    include_parked: bool,
) -> list[dict[str, str]]:
    open_rows: list[dict[str, str]] = []
    for r in rows:
        pid = r.get("problemId", "")
        if r.get("queue") not in queues:
            continue
        if r.get("practiceStatus") in SKIP_STATUS:
            continue
        if r.get("runReady") == "1":
            continue
        if pid in claimed:
            continue
        if not include_parked and pid in PARKED_PROBLEMS:
            continue
        if _policy_verified(pid, r.get("policyFile", "")):
            continue
        open_rows.append(r)

    def sort_key(r: dict[str, str]) -> tuple:
        pid = r["problemId"]
        state_ok = _state_exists(pid, r.get("stateFile", ""))
        pol_ok = _policy_exists(pid, r.get("policyFile", ""))
        # Fresh unstarted first (throughput); state_ready residual-stuck last.
        if r.get("practiceStatus") == "unstarted" and state_ok:
            band = 0
        elif r.get("practiceStatus") == "unstarted":
            band = 1
        else:
            band = 2  # state_ready residual chains
        # Prefer existing teleport fixture when band ties.
        teleport = 0 if (r.get("teleportReady") == "1" or state_ok) else 1
        if not prefer_teleport:
            teleport = 0
        # Prefer no policy yet (scaffold green path) over iterating red policy.
        has_pol = 1 if pol_ok else 0
        return (band, teleport, has_pol, int(r.get("rank") or 9999))

    open_rows.sort(key=sort_key)
    return open_rows[:count]


def _card_body(card_id: str, row: dict[str, str]) -> str:
    pid = row["problemId"]
    room = row["roomName"]
    rid = row["roomIdHex"]
    rank = row["rank"]
    status = row["practiceStatus"]
    state_file = row.get("stateFile", "")
    policy_file = row.get("policyFile", f"policies/room_clears/{pid}.json")
    state_on_disk = _state_exists(pid, state_file)
    pol_on_disk = _policy_exists(pid, policy_file)
    mode = "iterate" if state_on_disk else "bootstrap+scaffold"

    return f"""# TASK {card_id}: Dual-track room segment — {room}

## Recipe step
room practice segment (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/{pid}.json` (create or edit)
- entry fixture under `custom_integrations/SuperMetroid-Snes/` for **this
  problem only** (bootstrap/teleport state if missing)
- optional residual: `docs/tasks/{card_id}-residual.md`
- optional note: `docs/tasks/{card_id}-note.md`

Do **not** edit: `routes/continuous.py`, `docs/STATUS.md`, `routes/kpdr/*`,
`progression.py`, other rooms' policies, or any spine controller.

## Context
- Dual-track room farm (Wave 10+): continuous tip work is **parked**.
- One agent ↔ one problem — no cross-room edits (collision guard).
- Queue rank **{rank}**, room `{rid}` **{room}**, problem `{pid}`.
- Board practiceStatus: `{status}`; state_on_disk={state_on_disk};
  policy_on_disk={pol_on_disk}; mode=`{mode}`.
- Practice promote ≠ continuous integrity.

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `scripts/room/run_problem.py` (bootstrap / scaffold / teleport / run / promote)
- If residual exists for this problem, read the latest `docs/tasks/*-residual.md`
  or note mentioning `{pid}`.

## Do
1. If no teleport fixture: bootstrap this problem only:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py bootstrap {pid}
   uv run python super_metroid/scripts/room/run_problem.py teleport {pid}
   ```
2. Scaffold policy if missing:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py scaffold {pid}
   ```
3. Iterate isolated run until **green** or honest residual with pin:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run {pid}
   ```
4. Promote **only** on green isolated run (practice track):
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py run {pid} --promote
   ```
5. Write residual with PROCESS schema. Next card may be a one-knob residual
   for this same problem (`{card_id}-R1`) or `none` if green+promoted.

## Do not
- Claim continuous / STATUS green
- Touch another problem's policy or state
- Edit spine controllers or progression
- Forge progression/capacity/boss RAM for green claims
- Spend the session on open-ended exploration outside this problem

## Acceptance
- [ ] Isolated run **GREEN + promote** **or** honest residual with pin
- [ ] Only own-files touched
- [ ] Dual-track non-claim in residual
- [ ] Next card ID + one change filled

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport {pid}
uv run python super_metroid/scripts/room/run_problem.py run {pid}
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run {pid} --promote
```

## Done when
Residual filed (message and/or `{card_id}-residual.md`). Planner owns queue
refresh / continuous tip; this card never does continuous compose.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=8, help="how many new cards")
    ap.add_argument(
        "--queues",
        default="1,2",
        help="comma queues to include (default 1,2 = easy+standard)",
    )
    ap.add_argument(
        "--print-ids",
        action="store_true",
        help="print generated card ids one per line",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="select problems but do not write cards",
    )
    ap.add_argument(
        "--no-prefer-teleport",
        action="store_true",
        help="do not prefer rooms that already have teleport fixtures",
    )
    ap.add_argument(
        "--include-parked",
        action="store_true",
        help="include Wave-9 parked residual rooms (default: skip)",
    )
    args = ap.parse_args()
    queues = {q.strip() for q in args.queues.split(",") if q.strip()}

    if not QUEUE_CSV.is_file():
        print(f"missing queue: {QUEUE_CSV}", file=sys.stderr)
        return 2

    rows = _load_queue()
    claimed = _existing_card_problem_ids()
    selected = _select_rows(
        rows,
        count=args.count,
        queues=queues,
        claimed=claimed,
        prefer_teleport=not args.no_prefer_teleport,
        include_parked=args.include_parked,
    )
    if not selected:
        print("no open easy/standard problems left unclaimed", file=sys.stderr)
        return 1

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    start = _next_card_index()
    ids: list[str] = []
    for i, row in enumerate(selected):
        card_id = f"SM-ROOM-SEG-{start + i:02d}"
        body = _card_body(card_id, row)
        path = TASKS_DIR / f"{card_id}.md"
        if args.dry_run:
            print(
                f"[dry] {card_id} → {row['problemId']} ({row['roomName']}) "
                f"status={row['practiceStatus']}"
            )
        else:
            path.write_text(body, encoding="utf-8")
            print(
                f"wrote {path.relative_to(ROOT)} ← {row['problemId']} "
                f"({row['roomName']})"
            )
        ids.append(card_id)

    if args.print_ids:
        for i in ids:
            print(i)
    else:
        print("IDS:", " ".join(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
