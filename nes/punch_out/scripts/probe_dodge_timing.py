"""Micro-probe: after KD1, try timed dodge offsets vs first attack.

1. Replay taunt KD1 with left jabs.
2. Idle until next attack act enters ATTACK set.
3. At various frame offsets, pulse LEFT or RIGHT for N frames.
4. Report whether Mac took damage.

Also tests pure idle (block) and continuous hold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from punch_out.paths import GAME, GAME_DIR
from punch_out.policy import ATTACK_ACTS
from punch_out.ram import (
    ADDR_HEALTH,
    ADDR_OPP_ACTION,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    ADDR_OPP_PATTERN_TIMER,
    ADDR_KNOCKDOWN,
    is_match_live,
    is_taunt_window,
    hearts,
)
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless


def advance(env, action, n=1):
    obs = None
    for _ in range(n):
        step = env.step(action)
        obs = step[0] if isinstance(step, tuple) else step
    return obs


def reach_post_kd1(env):
    """From Match1 reset, taunt-counter to KD1, return when opp rises."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for _ in range(2000):
        if is_match_live(env.get_ram()):
            break
        advance(env, nes_idle_action())

    # Wait for strict taunt (pset==150)
    for _ in range(5000):
        ram = env.get_ram()
        if int(ram[ADDR_OPP_PATTERN_SET]) == 150:
            break
        advance(env, nes_idle_action())

    # Punch until KD
    for i in range(400):
        ram = env.get_ram()
        if int(ram[ADDR_OPP_HEALTH]) == 0:
            break
        if i % 5 < 2:
            advance(env, nes_action("A"))
        else:
            advance(env, nes_idle_action())

    # Wait for opp to rise (health > 0 again)
    for _ in range(2000):
        ram = env.get_ram()
        if int(ram[ADDR_OPP_HEALTH]) > 0 and int(ram[ADDR_KNOCKDOWN]) == 0:
            return True
        # dodge during count
        advance(env, nes_action("LEFT") if _ % 12 < 3 else nes_idle_action())
    return False


def wait_attack_start(env, max_frames=3000):
    """Idle until act enters attack set; return (pset, act, timer) snapshot."""
    prev = -1
    for f in range(max_frames):
        ram = env.get_ram()
        act = int(ram[ADDR_OPP_ACTION])
        pset = int(ram[ADDR_OPP_PATTERN_SET])
        timer = int(ram[ADDR_OPP_PATTERN_TIMER])
        if act in ATTACK_ACTS and prev not in ATTACK_ACTS:
            return f, pset, act, timer, int(ram[ADDR_HEALTH])
        if act in ATTACK_ACTS and prev in ATTACK_ACTS and act != prev:
            return f, pset, act, timer, int(ram[ADDR_HEALTH])
        prev = act
        advance(env, nes_idle_action())
    return None


def try_dodge_recipe(
    env,
    *,
    delay: int,
    side: str,
    hold: int,
    then_punch: int = 0,
    then_idle: int = 80,
) -> dict[str, Any]:
    """From current attack start: wait delay, hold side, optional punches."""
    mac0 = int(env.get_ram()[ADDR_HEALTH])
    opp0 = int(env.get_ram()[ADDR_OPP_HEALTH])
    # delay frames idle
    advance(env, nes_idle_action(), delay)
    # hold dodge
    if side == "IDLE":
        advance(env, nes_idle_action(), hold)
    else:
        advance(env, nes_action(side), hold)
    # optional counters
    for i in range(then_punch):
        if i % 5 < 2:
            advance(env, nes_action("A"))
        else:
            advance(env, nes_idle_action())
    advance(env, nes_idle_action(), then_idle)
    ram = env.get_ram()
    mac1 = int(ram[ADDR_HEALTH])
    opp1 = int(ram[ADDR_OPP_HEALTH])
    return {
        "mac0": mac0,
        "mac1": mac1,
        "mac_dmg": max(0, mac0 - mac1) if mac1 > 0 else mac0,
        "opp_dmg": max(0, opp0 - opp1) if opp1 >= 0 else 0,
        "mac1_raw": mac1,
        "opp1": opp1,
        "pset": int(ram[ADDR_OPP_PATTERN_SET]),
        "act": int(ram[ADDR_OPP_ACTION]),
    }


def main() -> None:
    configure_headless()
    # First: capture a save-state right at first post-KD1 attack start
    env = make_env(GAME, "Match1", GAME_DIR, render_mode="rgb_array")
    try:
        ok = reach_post_kd1(env)
        print(f"post_kd1 ok={ok} mac={int(env.get_ram()[ADDR_HEALTH])} "
              f"opp={int(env.get_ram()[ADDR_OPP_HEALTH])} hearts={hearts(env.get_ram())}")
        hit = wait_attack_start(env)
        if hit is None:
            print("no attack found")
            return
        f, pset, act, timer, mac = hit
        print(f"attack_start f+={f} pset={pset} act={act} timer={timer} mac={mac}")

        # Save state at attack start via emulator
        state_bytes = env.em.get_state()
        recipes = []
        # delays 0..40 step 2, sides L/R/IDLE, holds 2,3,5,8
        for delay in range(0, 45, 3):
            for side in ("LEFT", "RIGHT", "IDLE"):
                for hold in (2, 3, 5, 8):
                    recipes.append((delay, side, hold, 0))
        # also a few with counters
        for delay in (0, 5, 10, 15, 20, 30):
            for side in ("LEFT", "RIGHT"):
                recipes.append((delay, side, 5, 30))

        best = []
        for delay, side, hold, punch in recipes:
            env.em.set_state(state_bytes)
            # need one null step sometimes after load? try direct
            r = try_dodge_recipe(
                env, delay=delay, side=side, hold=hold, then_punch=punch, then_idle=60
            )
            r.update(delay=delay, side=side, hold=hold, punch=punch)
            best.append(r)

        # Sort by mac_dmg asc, then opp_dmg desc
        best.sort(key=lambda x: (x["mac_dmg"], -x["opp_dmg"]))
        print("\nBest 30 (lowest mac_dmg):")
        for r in best[:30]:
            print(
                f"  d={r['delay']:2d} {r['side']:5s} h={r['hold']} p={r['punch']:2d}  "
                f"mac_dmg={r['mac_dmg']:2d} opp_dmg={r['opp_dmg']:2d}  "
                f"mac {r['mac0']}->{r['mac1_raw']}"
            )
        print("\nZero mac_dmg recipes:")
        zeros = [r for r in best if r["mac_dmg"] == 0]
        print(f"  count={len(zeros)} / {len(best)}")
        for r in zeros[:20]:
            print(
                f"  d={r['delay']:2d} {r['side']:5s} h={r['hold']} p={r['punch']:2d}  "
                f"opp_dmg={r['opp_dmg']}"
            )
        print("\nAny opp damage:")
        for r in sorted(best, key=lambda x: -x["opp_dmg"])[:15]:
            if r["opp_dmg"] > 0:
                print(
                    f"  d={r['delay']:2d} {r['side']:5s} h={r['hold']} p={r['punch']:2d}  "
                    f"mac_dmg={r['mac_dmg']} opp_dmg={r['opp_dmg']}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
