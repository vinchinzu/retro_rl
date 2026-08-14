"""Experiment: try directional dodges and post-attack counters vs Glass Joe.

Runs short post-KD1 segments from Match1 with variants of dodge/counter
logic and reports mac damage + opp damage.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/punch_out/scripts/probe_counters.py
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from punch_out.paths import GAME, GAME_DIR
from punch_out.policy import ATTACK_ACTS
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_KNOCKDOWN,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    ADDR_OPP_PATTERN_TIMER,
    ADDR_ROUND,
    FIGHT_IN_RING,
    is_match_live,
    is_taunt_window,
    hearts,
    stars,
)
from retro_harness.env import make_env
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless

def _getup(g: int) -> FrameAction:
    if g % 6 in (0, 1):
        return FrameAction(nes_action("A"), "getup_a")
    if g % 6 in (3, 4):
        return FrameAction(nes_action("B"), "getup_b")
    return FrameAction(nes_idle_action(), "getup_idle")

@dataclass
class TrialStats:
    name: str
    frames: int = 0
    opp_kd: int = 0
    mac_kd: int = 0
    opp_dmg: int = 0
    mac_dmg: int = 0
    final_mac: int = 0
    final_opp: int = 0
    final_round: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    outcome: str = "timeout"

PolicyFn = Callable[[Any, int, dict], FrameAction]

def run_variant(
    name: str,
    choose: PolicyFn,
    *,
    max_frames: int = 20000,
) -> TrialStats:
    """Run one full bout with a custom choose(ram, t, state) function."""
    env = make_env(GAME, "Match1", GAME_DIR, render_mode="rgb_array")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for _ in range(2000):
            if is_match_live(env.get_ram()):
                break
            step = env.step(nes_idle_action())
            obs = step[0] if isinstance(step, tuple) else step

        stats = TrialStats(name=name)
        st: dict[str, Any] = {
            "phase": 0,
            "getup_g": 0,
            "mode": "fight",
            "prev_opp": 96,
            "prev_mac": 96,
            "prev_opp0": False,
            "counter_t": 0,
            "last_pset": 0,
            "last_act": 0,
        }
        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            mac = int(ram[ADDR_HEALTH])
            opp = int(ram[ADDR_OPP_HEALTH])
            fight = int(ram[ADDR_FIGHT_FLAG])
            opp0 = opp == 0

            if opp < st["prev_opp"]:
                stats.opp_dmg += st["prev_opp"] - opp
            if 0 < mac < st["prev_mac"]:
                stats.mac_dmg += st["prev_mac"] - mac
            if mac == 0 and st["prev_mac"] > 0:
                stats.mac_dmg += st["prev_mac"]
                stats.mac_kd += 1
                st["mode"] = "getup"
                st["getup_g"] = 0
            if mac > 0 and st["prev_mac"] == 0:
                st["mode"] = "fight"
            if opp0 and not st["prev_opp0"]:
                stats.opp_kd += 1
            st["prev_opp0"] = opp0
            st["prev_opp"] = opp
            st["prev_mac"] = mac

            if fight != FIGHT_IN_RING:
                fa = FrameAction(nes_action("A"), "between") if frame % 10 < 2 else FrameAction(
                    nes_idle_action(), "between_idle"
                )
            elif st["mode"] == "getup" or (mac == 0 and fight == FIGHT_IN_RING):
                fa = _getup(st["getup_g"])
                st["getup_g"] += 1
            else:
                fa = choose(ram, frame, st)

            stats.reasons[fa.reason] = stats.reasons.get(fa.reason, 0) + 1
            step = env.step(fa.action)
            obs = step[0] if isinstance(step, tuple) else step
            stats.frames = frame
            stats.final_mac = int(env.get_ram()[ADDR_HEALTH])
            stats.final_opp = int(env.get_ram()[ADDR_OPP_HEALTH])
            stats.final_round = int(env.get_ram()[ADDR_ROUND])

            if stats.mac_kd >= 3:
                stats.outcome = "loss_tko"
                break
            if stats.opp_kd >= 3:
                stats.outcome = "tko_win"
                break
            # KO: opp down long enough
            if opp0 and stats.opp_kd >= 1:
                # check knockdown count duration via state
                pass
        else:
            stats.outcome = "timeout"
            # Decision heuristic: higher opp_kd or lower opp health
            if stats.opp_kd > stats.mac_kd and stats.final_round >= 3:
                stats.outcome = "maybe_decision"
        return stats
    finally:
        env.close()

# --- Variant policies ---

def choose_hold_left(ram, t, st) -> FrameAction:
    """Always hold LEFT dodge."""
    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")
    return FrameAction(nes_action("LEFT"), "hold_left")

def choose_hold_right(ram, t, st) -> FrameAction:
    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")
    return FrameAction(nes_action("RIGHT"), "hold_right")

def choose_pulse_lr(ram, t, st) -> FrameAction:
    """Baseline 3/3 L/R pulse + taunt."""
    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")
    s = st["phase"] % 12
    st["phase"] += 1
    if s < 3:
        return FrameAction(nes_action("LEFT"), "left")
    if 6 <= s < 9:
        return FrameAction(nes_action("RIGHT"), "right")
    return FrameAction(nes_idle_action(), "idle")

def choose_block_only(ram, t, st) -> FrameAction:
    """Idle (block?) + taunt punch only."""
    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")
    return FrameAction(nes_idle_action(), "block")

def choose_timer_dodge(ram, t, st) -> FrameAction:
    """When timer low on attack acts, pulse dodge; else idle. Taunt punch."""
    pset = int(ram[ADDR_OPP_PATTERN_SET])
    act = int(ram[ADDR_OPP_ACTION])
    timer = int(ram[ADDR_OPP_PATTERN_TIMER])
    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")
    # Attack windup: dodge when timer is small
    if act in ATTACK_ACTS and timer <= 20:
        # Alternate side based on act parity
        side = "LEFT" if act % 2 == 1 else "RIGHT"
        st["phase"] += 1
        return FrameAction(nes_action(side), f"dodge_{side.lower()}")
    # After attack (timer just was low): counter punch briefly
    if st.get("counter_t", 0) > 0:
        st["counter_t"] -= 1
        if st["counter_t"] % 4 < 2:
            return FrameAction(nes_action("A"), "counter_a")
        return FrameAction(nes_idle_action(), "counter_rec")
    if act in ATTACK_ACTS and timer == 0:
        st["counter_t"] = 30
    return FrameAction(nes_idle_action(), "wait")

def choose_reactive_counter(ram, t, st) -> FrameAction:
    """Dodge on rising attack acts; counter with A/B after dodge window."""
    pset = int(ram[ADDR_OPP_PATTERN_SET])
    act = int(ram[ADDR_OPP_ACTION])
    timer = int(ram[ADDR_OPP_PATTERN_TIMER])
    prev_act = st.get("last_act", 0)
    st["last_act"] = act
    st["last_pset"] = pset

    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")

    # Enter counter window when attack act advances while timer small
    if act != prev_act and act in ATTACK_ACTS:
        st["counter_t"] = 0
        st["dodge_t"] = 8
        st["dodge_side"] = "LEFT" if act in (17, 23, 7, 10) else "RIGHT"
        st["after_dodge"] = 24

    if st.get("dodge_t", 0) > 0:
        st["dodge_t"] -= 1
        return FrameAction(nes_action(st.get("dodge_side", "LEFT")), "react_dodge")

    if st.get("after_dodge", 0) > 0:
        st["after_dodge"] -= 1
        # Counter punches: left face (A) then right face (B)
        n = st["after_dodge"]
        if n % 5 < 2:
            btn = "A" if (n // 5) % 2 == 0 else "B"
            return FrameAction(nes_action(btn), f"react_{btn.lower()}")
        return FrameAction(nes_idle_action(), "react_rec")

    # Opportunistic jabs when pattern looks open (115, 120 with low threat)
    if pset in (115, 120) and act in (0, 1, 2, 3, 8, 11, 13, 18, 21, 28) and timer > 30:
        if st["phase"] % 6 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "open_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "open_rec")

    return FrameAction(nes_idle_action(), "idle")

def choose_body_star_farm(ram, t, st) -> FrameAction:
    """Early left body blows + taunt; try star uppercut if stars available."""
    pset = int(ram[ADDR_OPP_PATTERN_SET])
    act = int(ram[ADDR_OPP_ACTION])
    timer = int(ram[ADDR_OPP_PATTERN_TIMER])
    s = stars(ram)
    h = hearts(ram)

    if is_taunt_window(ram):
        # Prefer star uppercut on taunt if available
        if s > 0:
            if st["phase"] % 8 < 3:
                st["phase"] += 1
                return FrameAction(nes_action("UP", "A"), "star_up")
            st["phase"] += 1
            return FrameAction(nes_idle_action(), "star_rec")
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")

    # Dodge attacks
    if act in ATTACK_ACTS and timer <= 25:
        side = "LEFT" if act in (17, 23, 7, 10, 6) else "RIGHT"
        st["phase"] += 1
        st["after_dodge"] = 20
        return FrameAction(nes_action(side), "dodge")

    if st.get("after_dodge", 0) > 0:
        st["after_dodge"] -= 1
        if st["after_dodge"] % 5 < 2:
            return FrameAction(nes_action("A"), "counter_a")
        return FrameAction(nes_idle_action(), "counter_rec")

    # Farm body blows early for stars when hearts allow
    if h > 0 and pset in (115, 120, 0) and act not in ATTACK_ACTS:
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("DOWN", "A"), "body_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "body_rec")

    return FrameAction(nes_idle_action(), "idle")

def choose_always_counter_left_jab(ram, t, st) -> FrameAction:
    """Classic: dodge then left jab spam after any attack; taunt with A."""
    act = int(ram[ADDR_OPP_ACTION])
    timer = int(ram[ADDR_OPP_PATTERN_TIMER])
    pset = int(ram[ADDR_OPP_PATTERN_SET])
    prev = st.get("last_act", -1)
    st["last_act"] = act

    if is_taunt_window(ram):
        if st["phase"] % 5 < 2:
            st["phase"] += 1
            return FrameAction(nes_action("A"), "taunt_a")
        st["phase"] += 1
        return FrameAction(nes_idle_action(), "taunt_rec")

    # Detect attack start: act enters attack set
    if act in ATTACK_ACTS and prev not in ATTACK_ACTS:
        st["dodge_t"] = 10
        # Right hook (~20?) dodge left; left jab dodge right
        st["dodge_side"] = "LEFT" if act in (20, 4, 6, 10) else "RIGHT"
        st["punch_t"] = 0

    if st.get("dodge_t", 0) > 0:
        st["dodge_t"] -= 1
        if st["dodge_t"] == 0:
            st["punch_t"] = 40
        return FrameAction(nes_action(st.get("dodge_side", "LEFT")), "dodge")

    if st.get("punch_t", 0) > 0:
        st["punch_t"] -= 1
        if st["punch_t"] % 5 < 2:
            return FrameAction(nes_action("A"), "jab")
        return FrameAction(nes_idle_action(), "jab_rec")

    # Idle block when quiet
    if pset in (115,) or (pset == 120 and act in (3, 8, 11, 13, 18, 21, 28)):
        return FrameAction(nes_idle_action(), "guard")

    # fallback short dodge pulse
    s = st["phase"] % 12
    st["phase"] += 1
    if s < 2:
        return FrameAction(nes_action("LEFT"), "fb_l")
    if 6 <= s < 8:
        return FrameAction(nes_action("RIGHT"), "fb_r")
    return FrameAction(nes_idle_action(), "fb_i")

VARIANTS: dict[str, PolicyFn] = {
    "pulse_lr": choose_pulse_lr,
    "hold_left": choose_hold_left,
    "hold_right": choose_hold_right,
    "block_only": choose_block_only,
    "timer_dodge": choose_timer_dodge,
    "reactive": choose_reactive_counter,
    "body_star": choose_body_star_farm,
    "counter_jab": choose_always_counter_left_jab,
}

def main() -> None:
    configure_headless()
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-frames", type=int, default=18000)
    args = p.parse_args()
    names = args.only or list(VARIANTS)
    print(f"{'name':12} {'out':12} f={''} okd mkd  odmg mdmg  mac opp rnd")
    results = []
    for name in names:
        st = run_variant(name, VARIANTS[name], max_frames=args.max_frames)
        results.append(st)
        print(
            f"{st.name:12} {st.outcome:12} f={st.frames:5d} "
            f"okd={st.opp_kd} mkd={st.mac_kd}  "
            f"odmg={st.opp_dmg:3d} mdmg={st.mac_dmg:3d}  "
            f"mac={st.final_mac:2d} opp={st.final_opp:2d} r{st.final_round}"
        )
        top = sorted(st.reasons.items(), key=lambda kv: -kv[1])[:8]
        print("   reasons:", ", ".join(f"{k}:{v}" for k, v in top))

if __name__ == "__main__":
    main()
