"""Screen-by-screen bill for the pre-L1 bomb errand: cost, kills, money, time.

``probe_contact.py`` answers "who hit Link" one contact at a time and
``bomb_budget.py`` answers "what should the corridor pay" with no emulator.
Neither answers the question the walk actually loses on: *which screen* spent
the health and *which screen* paid. A run total of "16 kills, 4 hits, 2
rupees" reads the same whether the damage was one screen or five.

Every column here is per screen, and damage is in 1/256 of a heart, because a
wooden chip is ``$0670 -= 0x80`` and never moves the whole-hearts byte at all.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_screen_tables.py --tag tables1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bomb_budget import ROWS, per_kill, row_of  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# The four octorok types, split the way the drop table splits them: red is
# ``Types0`` (31%, hearts and 1-rupees), blue is ``Types2`` (41%, bombs).
RED = {"octorok", "octorok_fast"}
BLUE = {"octorok_blue", "octorok_blue_fast"}
_NAME_TO_TYPE = {
    "octorok": 0x07, "octorok_fast": 0x08, "octorok_blue": 0x09,
    "octorok_blue_fast": 0x0A, "tektite_blue": 0x0D, "tektite": 0x0E,
    "leever_blue": 0x0F, "leever": 0x10, "peahat": 0x1A, "moblin": 0x04,
    "moblin_blue": 0x03, "zora": 0x11, "ghini": 0x21, "rope": 0x28,
}
ITEM = {"0x00": "bomb", "0x0f": "5R", "0x18": "1R", "0x21": "clock",
        "0x22": "heart", "0x23": "fairy"}


def _hearts(row: dict, key: str) -> str:
    return f"{row[key]:.2f}/{row['containers']}" if row["containers"] else "-"


def _tally(by_type: dict[str, int], names: set[str]) -> int:
    return sum(n for name, n in by_type.items() if name in names)


def _others(by_type: dict[str, int]) -> str:
    rest = {n: c for n, c in by_type.items() if n not in RED and n not in BLUE}
    return " ".join(f"{n}x{c}" for n, c in sorted(rest.items())) or "-"


def _drops(row: dict) -> str:
    got = {ITEM.get(k, k): v for k, v in row["drops_by_state"].items()}
    return " ".join(f"{k}x{v}" for k, v in sorted(got.items())) or "-"


def _causes(row: dict) -> str:
    return " ".join(f"{k}x{v}" for k, v in sorted(row["hits_by_cause"].items())) or "-"


def _rows_of(screen_row: dict) -> str:
    """ROM drop rows present in the wave, e.g. ``0`` or ``0,2``."""
    rows = {
        row_of(_NAME_TO_TYPE[name])
        for name in screen_row["spawned_by_type"]
        if name in _NAME_TO_TYPE
    }
    return ",".join(str(r) for r in sorted(rows)) or "-"


def table_bill(screens: list[dict]) -> str:
    head = (
        "| screen | frames | hunt f | hearts in -> out | damage | hits | cause"
        " | R | kills | spawned | streak | verdict |"
    )
    out = [head, "|" + "---|" * 12]
    for r in screens:
        if r.get("transit"):
            verdict = "transit"
        else:
            verdict = "cleared" if r["cleared"] else "open"
        out.append(
            f"| `{r['screen']}` | {r['frames']} | {r['hunt_frames']} | "
            f"{_hearts(r, 'hearts_in')} -> {_hearts(r, 'hearts_out')} | "
            f"{r['damage_hearts']:.2f} | {r['hits']} | {_causes(r)} | "
            f"{r['rupees']} | {r['kills']} | {r['spawned']} | "
            f"{r['streak_in']}->{r['streak_out']}"
            f"{' (' + str(r['streak_resets']) + ' reset)' if r['streak_resets'] else ''}"
            f" | {verdict} |"
        )
    return "\n".join(out)


def table_wave(screens: list[dict]) -> str:
    head = (
        "| screen | spawned red | spawned blue | other | killed red |"
        " killed blue | killed other | ROM row | drops | R |"
    )
    out = [head, "|" + "---|" * 10]
    for r in screens:
        sp, ki = r["spawned_by_type"], r["kills_by_type"]
        out.append(
            f"| `{r['screen']}` | {_tally(sp, RED)} | {_tally(sp, BLUE)} | "
            f"{_others(sp)} | {_tally(ki, RED)} | {_tally(ki, BLUE)} | "
            f"{sum(ki.values()) - _tally(ki, RED) - _tally(ki, BLUE)} | "
            f"{_rows_of(r)} | {_drops(r)} | {r['rupees']} |"
        )
    return "\n".join(out)


def table_drop_rate(screens: list[dict]) -> str:
    """Measured drop rate per ROM row against what ``DropItemRates`` bills.

    This is the corridor's open question: 53 kills for 7 drops is ~13%
    against rows billed at 31% and 59%.
    """
    killed: dict[int, int] = {}
    for r in screens:
        for name, n in r["kills_by_type"].items():
            type_id = _NAME_TO_TYPE.get(name)
            if type_id is not None:
                killed[row_of(type_id)] = killed.get(row_of(type_id), 0) + n
    drops = sum(sum(r["drops_by_state"].values()) for r in screens)
    out = [
        "| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill"
        " | kills measured | E[R] |",
        "|" + "---|" * 6,
    ]
    total_expected = 0.0
    for row, n in sorted(killed.items()):
        rate, _, baxter, loc, _ = ROWS[row]
        e_rupees, _ = per_kill(row)
        total_expected += e_rupees * n
        out.append(
            f"| {row} | {baxter} / {loc} | {rate / 256:.3f} | {e_rupees:.3f} |"
            f" {n} | {e_rupees * n:.2f} |"
        )
    total_kills = sum(killed.values())
    expected_drops = sum(ROWS[r][0] / 256 * n for r, n in killed.items())
    if not total_kills:
        out.append("\nno kills")
        return "\n".join(out)
    out.append(
        f"\n{total_kills} kills -> **{drops} floor drops** "
        f"({drops / total_kills:.0%} against {expected_drops / total_kills:.0%} "
        f"billed), E[R] {total_expected:.2f}."
    )
    return "\n".join(out)


def render(payload: dict) -> str:
    screens = payload["screens"]
    h = payload["hunt"]
    lines = [
        f"### Walk bill ({payload['tag']})",
        "",
        table_bill(screens),
        "",
        "### The wave, red against blue",
        "",
        table_wave(screens),
        "",
        "### Drop rate against the ROM rows",
        "",
        table_drop_rate(screens),
        "",
        "### Run totals",
        "",
        f"- kills {h['kills']} census / {h['kills_counter']} ROM counters",
        f"- rupees banked {h['rupees_banked']}, final {payload['final']['rupees']}",
        f"- damage {h['damage_hearts']:.2f} hearts over {h['damage_taken']} hits"
        f" ({h['hurt_events']} iframe arms)",
        f"- streak best {h['streak_best']}, resets {h['streak_resets']}",
        f"- hits by cause: {h['hits_by_cause']}",
        # What the value policy walked past, and what it never stood under.
        f"- prey passed on value: {h.get('prey_passed', {})}",
        f"- transit screens: {[hex(s) for s in h.get('transit_screens', [])]}"
        f" ({h.get('transit_frames', 0)}f)",
        f"- duck frames: {h.get('duck_frames', 0)}, shield {h.get('shield_frames', 0)}",
        f"- blade presses {h.get('blade_presses', 0)}, off-face"
        f" {h.get('blade_presses_off_face', 0)}, turn frames"
        f" {h.get('turn_frames', 0)}",
        "- stages: " + ", ".join(
            f"{s['name']} {s['frames']}f ok={s['ok']}" for s in payload["stages"]
        ),
        f"- last stage notes: {payload['stages'][-1]['notes']}",
        f"- result ok={payload['ok']} failed={payload['failed']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="screen_tables")
    # The blade's turn-before-swing rung, as an ablation. ``ScreenHunter`` is
    # built inside ``OverworldPathController``, so the field default is the
    # only seam a probe has (same trick as ``probe_contact.py``).
    parser.add_argument(
        "--turn-first", action=argparse.BooleanOptionalAction, default=None
    )
    # Extra screens to cross instead of clear, e.g. ``--transit 0x7c``.
    parser.add_argument("--transit", default=None)
    args = parser.parse_args(argv)
    configure_headless()

    if args.turn_first is not None:
        # Wrap ``__init__``, do *not* rewrite the dataclass field default:
        # ``@dataclass`` bakes each default into the generated ``__init__``
        # signature at class creation, so ``fields(...)[i].default = v`` and
        # ``setattr(cls, name, v)`` are both no-ops on every instance built
        # afterwards. ``probe_contact.py`` has been printing "overrides:"
        # over exactly that no-op.
        from zelda_i.overworld import hunt as hunt_mod

        want = bool(args.turn_first)
        _orig_init = hunt_mod.ScreenHunter.__init__

        def _init(self, *a, **kw):  # type: ignore[no-untyped-def]
            _orig_init(self, *a, **kw)
            self.turn_before_swing = want

        hunt_mod.ScreenHunter.__init__ = _init  # type: ignore[assignment]
        print("turn_before_swing:", want)

    if args.transit:
        from zelda_i.overworld import shop_p7 as shop_mod

        extra = frozenset(int(v, 0) for v in args.transit.split(","))
        shop_mod.SHOP_P7_TRANSIT_SCREENS = (
            shop_mod.SHOP_P7_TRANSIT_SCREENS | extra
        )
        print("transit:", sorted(hex(s) for s in shop_mod.SHOP_P7_TRANSIT_SCREENS))

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    payload: dict | None = None
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.tables"))
        run = run_survival_spine(
            env, obs, assist=None, through="pre-l1", allow_pokes=False
        )
        snap = read_snapshot(env.get_ram())
        report = run.report()
        hunt: dict = {}
        stages = []
        for stage in report.get("stages", []):
            ctl = stage.get("controller") or {}
            nested = ctl.get("hunt") if isinstance(ctl.get("hunt"), dict) else {}
            if "screens" in nested and stage.get("name") in (None, "bomb_walk"):
                hunt = nested
            elif "screens" in nested and not hunt:
                hunt = nested
            stages.append(
                {
                    "name": stage.get("name"),
                    "frames": stage.get("frames"),
                    "ok": stage.get("success"),
                    "phase": ctl.get("phase"),
                    "notes": (ctl.get("notes") or [])[-3:],
                }
            )
        payload = {
            "tag": args.tag,
            "ok": report.get("ok"),
            "failed": report.get("failed_stage"),
            "final": spine_final_fields(snap, env.get_ram()),
            "hunt": hunt,
            "screens": hunt.get("screens", []),
            "stages": stages,
        }
    finally:
        env.close()

    md = render(payload)
    write_json_report(OUT_DIR / f"{args.tag}.json", payload)
    write_json_report(RECORDINGS_DIR / f"{args.tag}.json", payload)
    (OUT_DIR / f"{args.tag}.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
