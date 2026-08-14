"""rr-zavx: Clean continuous L4 Entrance → skip-compass NaturalKey → TF 0x08.

Compose pure dual-green room segments from ``Level4Entrance`` (or mid
checkpoints) **without compass KEY-RIGHT** so keys≥1 remain for map KEY-UP,
then reuse ``run_level4_continuous_tf`` for map → Gleeok → ``tf&0x08``.

No ``--infinite-life`` for Clean claim. Not full-game power-on STATUS.

Examples::

    # Clean dual full compose (Entrance → TF)
    uv run python nes/zelda_i/scripts/run_level4_entrance_tf.py --trials 2 \\
        --save-state --tag l4_zavx_entrance_tf

    # Natural-key residual only (skip-compass → PostLadderNaturalKey)
    uv run python nes/zelda_i/scripts/run_level4_entrance_tf.py \\
        --to-natural-key-only --trials 2 --save-state --tag l4_zavx_natkey

    # From Room50Cleared (keys=1) through TF
    uv run python nes/zelda_i/scripts/run_level4_entrance_tf.py \\
        --from-state Level4Room50Cleared --trials 2 --tag l4_zavx_from50_tf
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, read_snapshot, read_u8
from zelda_i.scripts import run_level4_continuous_tf as cont_tf
from zelda_i.scripts import run_level4_rooms as r4

# Skip-compass natural-key spine (no key_right_62 / compass_62).
# Each tuple: (segment, default_start_if_chain_break, natural_checkpoint_name)
NATURAL_KEY_SPINE: tuple[tuple[str, str, str], ...] = (
    ("chain_to_key", "Level4Entrance", "Level4FirstKey"),
    ("clear_50", "Level4FirstKey", "Level4Room50Cleared"),
    ("north_40", "Level4Room50Cleared", "Level4Room40NaturalKey"),
    ("key_40", "Level4Room40NaturalKey", "Level4Room40ClearedNaturalKey"),
    ("north_30", "Level4Room40ClearedNaturalKey", "Level4Room30NaturalKey"),
    ("clear_30", "Level4Room30NaturalKey", "Level4Room30ClearedNaturalKey"),
    ("key_right_31", "Level4Room30ClearedNaturalKey", "Level4Room31NaturalKey"),
    ("clear_31", "Level4Room31NaturalKey", "Level4Room31ClearedNaturalKey"),
    ("east_32", "Level4Room31ClearedNaturalKey", "Level4Room32NaturalKey"),
    ("clear_32", "Level4Room32NaturalKey", "Level4Room32ClearedNaturalKey"),
    ("stepladder", "Level4Room32ClearedNaturalKey", "Level4StepladderNaturalKey"),
    ("exit_60", "Level4StepladderNaturalKey", "Level4PostLadderNaturalKey"),
    ("west_31", "Level4PostLadderNaturalKey", "Level4Room31PostLadderNaturalKey"),
)

# From-state shortcuts: skip spine segments already behind this state.
_FROM_STATE_SKIP_UNTIL: dict[str, str] = {
    "Level4Entrance": "chain_to_key",
    "Level4FirstKey": "clear_50",
    "Level4Room50Cleared": "north_40",
    "Level4Room40": "key_40",
    "Level4Room40NaturalKey": "key_40",
    "Level4Room40Cleared": "north_30",
    "Level4Room40ClearedNaturalKey": "north_30",
    "Level4Room30": "clear_30",
    "Level4Room30NaturalKey": "clear_30",
    "Level4Room30Cleared": "key_right_31",
    "Level4Room30ClearedNaturalKey": "key_right_31",
    "Level4Room31": "clear_31",
    "Level4Room31NaturalKey": "clear_31",
    "Level4Room31Cleared": "east_32",
    "Level4Room31ClearedNaturalKey": "east_32",
    "Level4Room32": "clear_32",
    "Level4Room32NaturalKey": "clear_32",
    "Level4Room32Cleared": "stepladder",
    "Level4Room32ClearedNaturalKey": "stepladder",
    "Level4Stepladder": "exit_60",
    "Level4StepladderNaturalKey": "exit_60",
    "Level4PostLadder": "west_31",
    "Level4PostLadderNaturalKey": "west_31",
    "Level4Room31PostLadderNaturalKey": "",  # spine done
    "Level4Room31PostLadder": "",
}

def _state_path(name: str) -> Path:
    return GAME_DIR / "custom_integrations" / GAME / f"{name}.state"

def _copy_checkpoint(src_name: str, dst_name: str) -> str | None:
    """Copy a just-saved segment checkpoint to the NaturalKey name."""
    src = _state_path(src_name)
    if not src.exists():
        return None
    dst = _state_path(dst_name)
    shutil.copy2(src, dst)
    return str(dst)

def _spine_from(start_state: str) -> list[tuple[str, str, str]]:
    first = _FROM_STATE_SKIP_UNTIL.get(start_state)
    if first is None:
        # Unknown: run full spine; first segment uses start_state override.
        return list(NATURAL_KEY_SPINE)
    if first == "":
        return []
    out: list[tuple[str, str, str]] = []
    seen = False
    for seg, default_start, ckpt in NATURAL_KEY_SPINE:
        if seg == first:
            seen = True
        if seen:
            out.append((seg, default_start, ckpt))
    return out

def run_natural_key_spine(
    *,
    start_state: str,
    infinite_life: bool,
    save_checkpoints: bool,
    tag: str,
    trial_i: int,
) -> dict[str, Any]:
    """Run skip-compass segments to Level4Room31PostLadderNaturalKey."""
    report: dict[str, Any] = {
        "ok": False,
        "bead": "rr-zavx",
        "start_state": start_state,
        "track": "assisted" if infinite_life else "clean",
        "trial": trial_i,
        "segments": [],
        "total_frames": 0,
    }
    spine = _spine_from(start_state)
    if not spine:
        # Already at PostLadderNaturalKey (or PostLadder).
        sp = _state_path(start_state)
        if not sp.exists():
            report["error"] = f"missing_state_{start_state}"
            return report
        # Ensure NaturalKey alias exists for continuous_tf default.
        if start_state != "Level4Room31PostLadderNaturalKey":
            if save_checkpoints:
                _copy_checkpoint(start_state, "Level4Room31PostLadderNaturalKey")
            # Verify keys≥1
            configure_headless()
            env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
            try:
                env.reset()
                snap = read_snapshot(env.get_ram())
                lad = int(read_u8(env.get_ram(), ADDR_LADDER))
                report["final"] = {
                    "room": snap.screen,
                    "room_hex": f"0x{snap.screen:02x}",
                    "keys": snap.keys,
                    "ladder": lad,
                    "health": snap.health,
                }
                if lad <= 0:
                    report["error"] = "no_ladder"
                    return report
                if snap.keys < 1:
                    report["error"] = "no_keys_natural_key_required"
                    return report
                report["ok"] = True
                report["checkpoint"] = "Level4Room31PostLadderNaturalKey"
                return report
            finally:
                env.close()
        report["ok"] = True
        report["checkpoint"] = start_state
        return report

    current = start_state
    for i, (segment, default_start, nat_ckpt) in enumerate(spine):
        # First segment may use caller's start_state; later use prior checkpoint.
        if i == 0:
            seg_start = start_state if _state_path(start_state).exists() else default_start
        else:
            seg_start = current

        seg_tag = f"{tag}_t{trial_i}_{segment}"
        # Always save mid-spine so the next segment can reload this trial's
        # pose (chain is checkpoint-mediated). NaturalKey aliases preserved.
        r = r4.run_once(
            segment=segment,
            start_state=seg_start,
            infinite_life=infinite_life,
            save_checkpoint=True,
            tag=seg_tag,
            allow_key_poke=False,
        )
        entry = {
            "segment": segment,
            "start": seg_start,
            "ok": r.get("ok"),
            "error": r.get("error"),
            "frames": r.get("frames"),
            "final": r.get("final"),
            "checkpoint": r.get("checkpoint"),
        }
        report["segments"].append(entry)
        report["total_frames"] += int(r.get("frames") or 0)
        if not r.get("ok"):
            report["error"] = f"{segment}:{r.get('error') or 'failed'}"
            report["failed_segment"] = segment
            return report

        # Promote to NaturalKey-named checkpoint when different from default.
        default_ckpt = r4._CHECKPOINT.get(segment)  # noqa: SLF001
        if default_ckpt and nat_ckpt != default_ckpt:
            copied = _copy_checkpoint(default_ckpt, nat_ckpt)
            entry["natural_checkpoint"] = copied
            # provenance sidecar
            if copied:
                write_state_provenance(
                    Path(copied),
                    source_state_path=_state_path(seg_start),
                    request={
                        "bead": "rr-zavx",
                        "segment": f"l4_{segment}_natural_key",
                        "track": report["track"],
                        "skip_compass": True,
                    },
                    selected_trial={
                        "ok": True,
                        "segment": segment,
                        "frames": r.get("frames"),
                        "final": r.get("final"),
                    },
                    natural_entry=False,
                )
        current = nat_ckpt if _state_path(nat_ckpt).exists() else (
            default_ckpt or seg_start
        )

    # Final must be PostLadderNaturalKey with keys≥1 and ladder.
    final_state = "Level4Room31PostLadderNaturalKey"
    if not _state_path(final_state).exists():
        # fall back to default west_31 checkpoint + copy
        if _state_path("Level4Room31PostLadder").exists():
            _copy_checkpoint("Level4Room31PostLadder", final_state)
        else:
            report["error"] = "missing_postladder_natural_key"
            return report

    configure_headless()
    env = make_env(GAME, final_state, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        snap = read_snapshot(env.get_ram())
        lad = int(read_u8(env.get_ram(), ADDR_LADDER))
        report["final"] = {
            "room": snap.screen,
            "room_hex": f"0x{snap.screen:02x}",
            "keys": snap.keys,
            "ladder": lad,
            "health": snap.health,
            "mode": snap.mode,
        }
        if lad <= 0:
            report["error"] = "no_ladder_at_postladder"
            return report
        if snap.keys < 1:
            report["error"] = "no_keys_at_postladder_natural"
            return report
        report["ok"] = True
        report["checkpoint"] = final_state
        return report
    finally:
        env.close()

def run_once(
    *,
    start_state: str,
    infinite_life: bool,
    save_checkpoints: bool,
    tag: str,
    trial_i: int,
    to_natural_key_only: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "bead": "rr-zavx",
        "start_state": start_state,
        "track": "assisted" if infinite_life else "clean",
        "trial": trial_i,
        "tag": tag,
        "key_poke": False,
        "natural_entry": False,
        "compose": "entrance_skip_compass_to_tf",
    }

    spine_rep = run_natural_key_spine(
        start_state=start_state,
        infinite_life=infinite_life,
        save_checkpoints=save_checkpoints,
        tag=tag,
        trial_i=trial_i,
    )
    report["spine"] = {
        "ok": spine_rep.get("ok"),
        "error": spine_rep.get("error"),
        "frames": spine_rep.get("total_frames"),
        "segments": [
            {
                "segment": s.get("segment"),
                "ok": s.get("ok"),
                "frames": s.get("frames"),
                "error": s.get("error"),
                "keys": (s.get("final") or {}).get("keys"),
                "room": (s.get("final") or {}).get("room_hex"),
            }
            for s in (spine_rep.get("segments") or [])
        ],
        "final": spine_rep.get("final"),
        "checkpoint": spine_rep.get("checkpoint"),
    }
    if not spine_rep.get("ok"):
        report["error"] = f"spine:{spine_rep.get('error')}"
        report["total_frames"] = spine_rep.get("total_frames")
        return report

    if to_natural_key_only:
        report["ok"] = True
        report["tf08"] = False
        report["total_frames"] = spine_rep.get("total_frames")
        report["final"] = spine_rep.get("final")
        report["checkpoint"] = spine_rep.get("checkpoint")
        return report

    # Phase 2: continuous PostLadderNaturalKey → map → Gleeok → TF (Clean).
    cont = cont_tf.run_once(
        start_state="Level4Room31PostLadderNaturalKey",
        infinite_life=infinite_life,
        save_checkpoint=save_checkpoints and trial_i == 0,
        tag=f"{tag}_cont_t{trial_i}",
        trial_i=trial_i,
        from_map=False,
    )
    report["continuous_tf"] = {
        "ok": cont.get("ok"),
        "tf08": cont.get("tf08"),
        "error": cont.get("error"),
        "total_frames": cont.get("total_frames"),
        "key_poke": cont.get("key_poke"),
        "map_ok": (cont.get("map") or {}).get("ok"),
        "final": cont.get("final"),
    }
    if cont.get("key_poke"):
        report["key_poke"] = True
        report["error"] = "unexpected_key_poke"
        return report
    report["ok"] = bool(cont.get("ok") and cont.get("tf08"))
    report["tf08"] = bool(cont.get("tf08"))
    report["total_frames"] = int(spine_rep.get("total_frames") or 0) + int(
        cont.get("total_frames") or 0
    )
    report["final"] = cont.get("final") or spine_rep.get("final")
    if not report["ok"]:
        report["error"] = cont.get("error") or (cont.get("gleeok_path") or {}).get(
            "error"
        )
    return report

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level4Entrance")
    p.add_argument(
        "--to-natural-key-only",
        action="store_true",
        help="Stop at Level4Room31PostLadderNaturalKey (no map/Gleeok/TF)",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l4_zavx_entrance_tf")
    args = p.parse_args()

    trials: list[dict[str, Any]] = []
    for i in range(args.trials):
        print(f"=== trial {i} from {args.from_state} ===", flush=True)
        r = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoints=args.save_state,
            tag=args.tag,
            trial_i=i,
            to_natural_key_only=args.to_natural_key_only,
        )
        print(
            "RESULT",
            {
                "ok": r.get("ok"),
                "tf08": r.get("tf08"),
                "error": r.get("error"),
                "frames": r.get("total_frames"),
                "spine_ok": (r.get("spine") or {}).get("ok"),
                "key_poke": r.get("key_poke"),
                "keys_postladder": ((r.get("spine") or {}).get("final") or {}).get(
                    "keys"
                ),
            },
            flush=True,
        )
        trials.append(r)

    dual = (
        all(t.get("ok") for t in trials)
        and len(trials) >= 2
        and (
            args.to_natural_key_only
            or all(t.get("tf08") for t in trials)
        )
    )
    out = {
        "bead": "rr-zavx",
        "segment": (
            "continuous_entrance_skip_compass_natural_key"
            if args.to_natural_key_only
            else "continuous_entrance_skip_compass_tf"
        ),
        "from": args.from_state,
        "to_natural_key_only": args.to_natural_key_only,
        "dual_green": dual,
        "ok": dual or (len(trials) == 1 and trials[0].get("ok")),
        "track": "assisted" if args.infinite_life else "clean",
        "key_poke": any(t.get("key_poke") for t in trials),
        "trials": trials,
        "tag": args.tag,
        "note": (
            "Lab checkpoint continuous from Level4Entrance; not full-game "
            "power-on STATUS. Skip-compass pure spine + Clean PostLadder→TF."
        ),
    }
    path = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(path, out)
    print(f"wrote {path} dual={dual} ok={out['ok']}", flush=True)
    return 0 if out["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
