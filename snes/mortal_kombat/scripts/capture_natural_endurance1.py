#!/usr/bin/env python3
"""Capture a model-free Endurance 1 continuation from the natural Fight 7 pin.

Replays NATURAL_THROUGH_FIGHT7_RLE from power-on, snapshots the emulator,
identifies the live E1 opponent (leftover HUD is ignored), then drives both
endurance bouts with RAM specialists first. Pixel speedrun / ladder-ft are
fallbacks after RAM oracles miss. Runtime artifacts are RLE only.

Natural E1 from the Fight 7 pin is courtyard Kano, still best-of-3 with
health refill between rounds. The second fighter appears only after Kano
is beaten two rounds (match_counter 7→8); leftover Match 7 HUD is not
that swap. Tournament ``ladder_model`` only covers M1–M7, so this
capture forces the oracle onto E1 and E1B slots as well. Match5 v3 can
take round 1 and then lose 1-2; ``--round2-kano`` switches to a no-jump
keepaway after that first KO. Keepaway ducks when knife sprite 0x1B36
leaves Kano — ``p2.state`` stays 0 for the whole throw. Courtyard
specialist (``scripted-courtyard``) idles so Kano commits the knife,
jump-forwards at 240f after a *visible* fight (296f from first-ready
black fade), then land HK. Ducking or jumping early makes Kano rush.

Liu Kang CPU walkthrough (IceMaster / LWang): fireball F,F,HP and flying
kick F,F,HK. Jump-kick into flying kick; fireball on wakeup. Do not jump
into Sub-Zero ice or Scorpion spear (Cage shadow kick is the same trap).
Leftover pin HUD still shows the Liu Kang mirror. Wait for a visible
fight frame — first fight-ready can still be a black fade.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "snes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.snapshot import get_emulator_state, set_emulator_state  # noqa: E402
from mortal_kombat.boot import BootController, action_from_buttons  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    ADDR_MATCH_COUNTER,
    ADDR_P1_X,
    ADDR_P1_Y,
    ADDR_P2_X,
    LIU_KANG_ID,
    PUNCH_RANGE,
    Screen,
    char_name,
    is_fight_ready,
    parse_ram,
)
from mortal_kombat.scripted import (  # noqa: E402
    B,
    DOWN,
    FIREBALL_RANGE,
    ScriptedPolicy,
    UP,
    X,
    back,
    fireball_sequence,
    flying_kick_sequence,
    forward,
    zeros,
)
from mortal_kombat.roster import KIND_RAM_V3, KIND_SCRIPT  # noqa: E402
from mortal_kombat.scripts.replay_natural_fight1 import buttons_from_mask  # noqa: E402
from mortal_kombat.scripts.replay_natural_fight7 import (  # noqa: E402
    NATURAL_THROUGH_FIGHT7_FRAMES,
    NATURAL_THROUGH_FIGHT7_RLE,
)
from dataclasses import replace  # noqa: E402

from mortal_kombat.tournament import TournamentRunner  # noqa: E402

ENDURANCE1 = 7
ENDURANCE1B = 8
ENDURANCE2 = 9


def mask_from_buttons(buttons: np.ndarray) -> int:
    mask = 0
    for index, value in enumerate(np.asarray(buttons).reshape(-1)[:12]):
        if int(value):
            mask |= 1 << index
    return mask


def rle_encode(masks: list[int]) -> list[tuple[int, int]]:
    encoded: list[tuple[int, int]] = []
    for mask in masks:
        if encoded and encoded[-1][0] == mask:
            encoded[-1] = (mask, encoded[-1][1] + 1)
        else:
            encoded.append((mask, 1))
    return encoded


def format_rle(pairs: list[tuple[int, int]], width: int = 8) -> str:
    chunks = [f"({mask}, {count})" for mask, count in pairs]
    lines = ["NATURAL_ENDURANCE1_RLE: tuple[tuple[int, int], ...] = ("]
    for start in range(0, len(chunks), width):
        lines.append("    " + ", ".join(chunks[start : start + width]) + ",")
    lines.append(")")
    return "\n".join(lines)


class RecordingEnv:
    """Record 12-button masks while forwarding to a live retro env."""

    def __init__(self, env):
        self.env = env
        self.masks: list[int] = []

    def step(self, action):
        self.masks.append(mask_from_buttons(action))
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def replay_through_fight7(env) -> None:
    reset_obs(env)
    frame = 0
    for mask, count in NATURAL_THROUGH_FIGHT7_RLE:
        buttons = buttons_from_mask(mask)
        for _ in range(count):
            env.step(buttons)
            frame += 1
            if frame % 5000 == 0:
                snap = parse_ram(env.unwrapped.get_ram())
                print(
                    f"  replay f={frame}/{NATURAL_THROUGH_FIGHT7_FRAMES} "
                    f"{describe(snap)}",
                    flush=True,
                )


def describe(snap) -> str:
    return (
        f"screen={snap.screen.name} match={snap.match_counter} "
        f"char={snap.p1_character}/{char_name(snap.p1_character)} "
        f"p2={snap.p2_character}/{char_name(snap.p2_character)} "
        f"hp={snap.p1_health}/{snap.p2_health} "
        f"rounds={snap.p1_rounds}-{snap.p2_rounds} timer={snap.timer}"
    )


def save_rgb(env, path: Path) -> None:
    rgb = env.render()
    if rgb is None:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb)).save(path)
    print(f"wrote {path} mean={float(np.mean(rgb)):.1f}")


def rgb_mean(env) -> float:
    rgb = env.render()
    if rgb is None:
        return 0.0
    return float(np.mean(rgb))


def identify_live_endurance1(env, pin, *, max_frames: int, screenshot: Path | None):
    """Idle through VS/load until Endurance 1 is fight-ready and on-screen.

    First fight-ready can still be a black fade with leftover/default p2.
    Wait for a visible frame so leftover HUD is not the opponent.
    """
    set_emulator_state(env.unwrapped, pin)
    boot = BootController(allow_continue=False)
    last = parse_ram(env.unwrapped.get_ram())
    print(f"identify pin {describe(last)}", flush=True)
    ready_at: int | None = None
    already = last.match_counter == ENDURANCE1 and is_fight_ready(last)
    for frame in range(0 if already else 1, max_frames + 1):
        if frame > 0:
            _phase, names = boot.decide(last, frame)
            env.step(action_from_buttons(names))
            last = parse_ram(env.unwrapped.get_ram())
        if frame % 200 == 0:
            print(
                f"  identify f={frame} mean={rgb_mean(env):.1f} {describe(last)}",
                flush=True,
            )
        if last.match_counter == ENDURANCE1 and is_fight_ready(last):
            if ready_at is None:
                ready_at = frame
                print(
                    f"  first-ready f={frame} mean={rgb_mean(env):.1f} {describe(last)}",
                    flush=True,
                )
            if rgb_mean(env) >= 8.0:
                if screenshot is not None:
                    save_rgb(env, screenshot)
                return frame, last
        if last.screen is Screen.CONTINUE:
            break
    if screenshot is not None:
        save_rgb(env, screenshot)
    return max_frames, last


# Sprite X that sticks to Kano until the knife leaves. Probed idle from the
# visible Fight 7 pin: 180 (== p2_x) through f=250, then 139/98/74 into P1.
ADDR_KNIFE_X = 0x1B36
KNIFE_SPLIT = 12


def knife_incoming(ram, p1_x: int, p2_x: int) -> bool:
    """True when 0x1B36 has left Kano and is still between the bodies.

    No slack past the bodies: when Kano walks off 180 the sprite can sit
    stale at 180, which is *behind* him, not a knife.
    """
    if ADDR_KNIFE_X >= len(ram):
        return False
    knife_x = int(ram[ADDR_KNIFE_X]) & 0xFF
    if abs(knife_x - p2_x) < KNIFE_SPLIT:
        return False
    lo, hi = (p1_x, p2_x) if p1_x <= p2_x else (p2_x, p1_x)
    return lo < knife_x < hi


class NoJumpFireballPolicy(ScriptedPolicy):
    """Courtyard Kano: fireball from far / duck knives / never jump.

    F,F,HP walks forward ~8f. Fire only when distance still leaves space
    after that walk so the knife does not eat the startup. ``p2.state``
    stays 0 for the whole knife; duck when 0x1B36 leaves Kano, cancel a
    started fireball, and walk back on cooldown instead of standing still.
    """

    name = "scripted-kano"
    fire_min = FIREBALL_RANGE + 24  # 96
    zone = FIREBALL_RANGE + 28  # 100
    duck_range = FIREBALL_RANGE
    knife_hold = 8
    corner_x = 32

    def __init__(self, intro_frames: int = 0) -> None:
        super().__init__(intro_frames=intro_frames)
        self._knife = 0

    def reset(self) -> None:
        super().reset()
        self._knife = 0

    def _pose(self, ram):
        p1_x = int(ram[ADDR_P1_X]) & 0xFF if ADDR_P1_X < len(ram) else 0
        p2_x = int(ram[ADDR_P2_X]) & 0xFF if ADDR_P2_X < len(ram) else 0
        return p1_x, p2_x, abs(p2_x - p1_x)

    def act(self, ram, rgb, *, deterministic: bool = False):
        snap = parse_ram(ram)
        p1_x, p2_x, _dist = self._pose(ram)
        if snap.screen is Screen.FIGHT and (
            snap.p2.state != 0 or knife_incoming(ram, p1_x, p2_x)
        ):
            self._knife = self.knife_hold
            self._queue.clear()
        return super().act(ram, rgb, deterministic=deterministic)

    def _duck(self, dist: int) -> np.ndarray:
        protect = zeros()
        protect[DOWN] = 1
        if dist <= PUNCH_RANGE:
            protect[X] = 1
        return protect

    def _choose(self, snap, ram):
        if snap.screen is not Screen.FIGHT:
            return zeros()
        full = snap.p1_health == snap.p2_health == 161
        if full and not self._was_full:
            self._intro = 0
        self._was_full = full
        if snap.p1_health < self._prev_hp:
            self._hurt = 24
        self._prev_hp = snap.p1_health
        if self._intro < self.intro_frames:
            self._intro += 1
            return zeros()
        if snap.p1.state != 0:
            return zeros()
        p1_x, p2_x, dist = self._pose(ram)
        facing = 1 if p1_x <= p2_x else -1
        if snap.p2.state != 0 or knife_incoming(ram, p1_x, p2_x):
            self._knife = self.knife_hold
        if self._knife > 0:
            self._knife -= 1
            return self._duck(dist)
        if self._hurt > 0:
            self._hurt -= 1
            return self._duck(dist) if dist <= self.duck_range else back(facing)
        cornered = p1_x < self.corner_x or p1_x > 255 - self.corner_x
        if dist < self.zone and not cornered:
            return back(facing)
        if self._cooldown > 0:
            return back(facing) if not cornered else self._duck(dist)
        if dist >= self.fire_min or cornered:
            return self._enqueue(fireball_sequence(facing))
        return back(facing)


class CourtyardKanoPolicy(ScriptedPolicy):
    """Jump the committed courtyard knife, land HK, then flying-kick.

    Intro eats jumps until ~240f after a *visible* fight. First-ready is
    a 51f black fade with pose already 68/144, so round 1 waits 296f
    from 161/161. Duck or jump before that and Kano cancels the knife
    and rushes. Jump on the in-flight sprite is too late.
    """

    name = "scripted-courtyard"
    round1_jump_at = 296
    later_jump_at = 240
    jump_hold = 10
    jump_startup = 40
    hk_hold = 8
    kick_range = FIREBALL_RANGE + 28  # 100

    def __init__(
        self,
        round1_jump_at: int = 296,
        later_jump_at: int = 240,
    ) -> None:
        super().__init__(intro_frames=0)
        self.round1_jump_at = round1_jump_at
        self.later_jump_at = later_jump_at
        self._fights = 0
        self._clock = 0
        self._airborne = False
        self._hk_left = 0
        self._opener_done = False

    def reset(self) -> None:
        super().reset()
        self._clock = 0
        self._airborne = False
        self._hk_left = 0
        self._opener_done = False

    @property
    def jump_at(self) -> int:
        return self.round1_jump_at if self._fights <= 1 else self.later_jump_at

    def act(self, ram, rgb, *, deterministic: bool = False):
        snap = parse_ram(ram)
        live = (
            snap.screen is Screen.FIGHT
            and snap.p1_health > 0
            and snap.p2_health > 0
            and snap.timer > 50
        )
        p1_x = int(ram[ADDR_P1_X]) & 0xFF if ADDR_P1_X < len(ram) else 0
        p2_x = int(ram[ADDR_P2_X]) & 0xFF if ADDR_P2_X < len(ram) else 0
        # Leftover Match 7 KO is screen=FIGHT hp=59/0; do not duck that.
        if live and knife_incoming(ram, p1_x, p2_x):
            self._queue.clear()
            protect = zeros()
            protect[DOWN] = 1
            if abs(p2_x - p1_x) <= PUNCH_RANGE:
                protect[X] = 1
            return protect
        if not live:
            self._queue.clear()
            return zeros()
        if self._queue:
            return self._queue.popleft()
        if self._cooldown > 0:
            self._cooldown -= 1
        return self._choose(snap, ram)

    def _choose(self, snap, ram):
        if snap.screen is not Screen.FIGHT:
            return zeros()
        full = snap.p1_health == snap.p2_health == 161
        if full and not self._was_full:
            self._clock = 0
            self._airborne = False
            self._hk_left = 0
            self._opener_done = False
            self._fights += 1
            print(
                f"  courtyard fight#{self._fights} jump_at={self.jump_at}",
                flush=True,
            )
        self._was_full = full
        if snap.p1_health > 0 and snap.p2_health > 0 and snap.timer > 50:
            self._clock += 1
        p1_x = int(ram[ADDR_P1_X]) & 0xFF if ADDR_P1_X < len(ram) else 0
        p1_y = int(ram[ADDR_P1_Y]) & 0xFF if ADDR_P1_Y < len(ram) else 0
        p2_x = int(ram[ADDR_P2_X]) & 0xFF if ADDR_P2_X < len(ram) else 0
        dist = abs(p2_x - p1_x)
        facing = 1 if p1_x <= p2_x else -1
        if 40 < p1_y < 140:
            self._airborne = True
            return zeros()
        if self._airborne and not self._opener_done:
            self._opener_done = True
            self._hk_left = self.hk_hold
            print(
                f"  courtyard land-hk clock={self._clock} x={p1_x}/{p2_x} "
                f"y={p1_y} dist={dist}",
                flush=True,
            )
        if not self._opener_done:
            if self.jump_at <= self._clock < self.jump_at + self.jump_hold:
                if self._clock == self.jump_at:
                    print(
                        f"  courtyard jump clock={self._clock} x={p1_x}/{p2_x} y={p1_y}",
                        flush=True,
                    )
                out = forward(facing)
                out[UP] = 1
                return out
            if self._clock < self.jump_at:
                return zeros()
            # 10f tap commits the jump; y drops ~20-30f later. Do not
            # flying-kick or walk during startup — that cancels it.
            if self._clock < self.jump_at + self.jump_startup:
                return zeros()
            print(
                f"  courtyard jump missed clock={self._clock} y={p1_y}",
                flush=True,
            )
            self._opener_done = True
        if self._hk_left > 0:
            self._hk_left -= 1
            out = zeros()
            out[B] = 1
            return out
        if snap.p1.state != 0:
            return zeros()
        if self._cooldown == 0 and dist <= self.kick_range:
            return self._enqueue(flying_kick_sequence(facing))
        if self._cooldown == 0 and dist > self.kick_range:
            return self._enqueue(fireball_sequence(facing))
        return forward(facing)


class RoundMixPolicy:
    """Use ``first`` until Kano is KO'd once, then ``rest`` on the refill.

    Match5 v3 can take courtyard Kano round 1 and then lose 1-2. The live
    game is unchanged; recorded buttons stay a Clean RLE tape.
    """

    def __init__(self, first, rest):
        self.first = first
        self.rest = rest
        self.kind = getattr(first, "kind", KIND_RAM_V3)
        self.name = f"{getattr(first, 'name', 'first')}+{getattr(rest, 'name', 'rest')}"
        self._koed = False
        self._armed = False
        self._active = first

    def reset(self) -> None:
        # TournamentRunner resets on every entered_fight. Courtyard HUD
        # p1_rounds lags, so clearing _koed would drop round 2 back onto
        # the round-1 oracle. A new mix object is built per capture attempt.
        self.first.reset()
        self.rest.reset()
        self._active = self.rest if self._koed else self.first

    def act(self, ram, rgb, *, deterministic: bool = False):
        snap = parse_ram(ram)
        # Pin leftover is Match 7's hp=59/0. Do not treat that as a Kano KO.
        if snap.timer > 50 and snap.p1_health > 0 and snap.p2_health > 0:
            self._armed = True
        if self._armed and snap.p2_health == 0 and snap.p1_health > 0:
            self._koed = True
        # Stay on rest after the first live KO. Leftover Match 7 HUD is
        # rounds=2-0 / hp=59/0 and must not count; _armed is the gate.
        policy = self.rest if self._koed else self.first
        if policy is not self._active:
            print(
                f"  mix switch -> {getattr(policy, 'name', '?')} "
                f"koed={self._koed} rounds={snap.p1_rounds}-{snap.p2_rounds} "
                f"hp={snap.p1_health}/{snap.p2_health}",
                flush=True,
            )
            policy.reset()
            self._active = policy
        return policy.act(ram, rgb, deterministic=deterministic)


class RelabelMatchPolicy:
    """Offline oracle: lie about match_counter in the policy's RAM copy.

    The live game is unchanged. Recorded buttons stay a Clean RLE tape.
    Match5 v3 was trained at match_id 4 vs Kano; natural E1 is match_id 7.
    """

    def __init__(self, inner, match_counter: int):
        self.inner = inner
        self.kind = getattr(inner, "kind", KIND_RAM_V3)
        self.name = getattr(inner, "name", "")
        self.match_counter = match_counter

    def reset(self) -> None:
        self.inner.reset()

    def act(self, ram, rgb, *, deterministic: bool = False):
        poked = np.array(ram, copy=True)
        if ADDR_MATCH_COUNTER < len(poked):
            poked[ADDR_MATCH_COUNTER] = self.match_counter & 0xFF
        return self.inner.act(poked, rgb, deterministic=deterministic)


def make_policy_loader(
    relabel_match: int | None,
    *,
    kano_script: bool = False,
    courtyard: bool = False,
    round2_kano: bool = False,
):
    def loader(path, kind):
        from mortal_kombat.compat import install_fighters_common_alias
        from mortal_kombat.policy import load_policy

        install_fighters_common_alias()
        if kind == KIND_SCRIPT and courtyard:
            return CourtyardKanoPolicy()
        if kind == KIND_SCRIPT and kano_script:
            return NoJumpFireballPolicy()
        policy = load_policy(path, kind)
        if relabel_match is not None and kind == KIND_RAM_V3:
            policy = RelabelMatchPolicy(policy, relabel_match)
        if round2_kano and not kano_script and not courtyard:
            # Round-2 keepaway must not stand still for the round-1 intro.
            return RoundMixPolicy(policy, NoJumpFireballPolicy(intro_frames=0))
        return policy

    return loader


def apply_oracle(runner: TournamentRunner, *, ladder_model: str | None, pixel_model: str | None) -> None:
    """Force the chosen oracle onto every slot, including E1/E1B.

    TournamentRunner.ladder_model only rewrites match_id <= 6 (M1–M7).
    Endurance capture needs the same zip on match_id 7 and 8.
    """
    if pixel_model:
        runner.slots = [
            replace(slot, model=pixel_model, kind="pixel", backups=[])
            for slot in runner.slots
        ]
        return
    if ladder_model:
        runner.slots = [
            replace(slot, model=ladder_model, kind=KIND_RAM_V3, backups=[])
            for slot in runner.slots
        ]


def capture_from_pin(
    env,
    pin,
    *,
    ladder_model: str | None,
    force_scripted: bool,
    max_frames: int,
    pixel_model: str | None = None,
    deterministic: bool = True,
    win_at: int = ENDURANCE2,
    relabel_match: int | None = None,
    kano_script: bool = False,
    courtyard: bool = False,
    round2_kano: bool = False,
) -> tuple[bool, list[int], object]:
    set_emulator_state(env.unwrapped, pin)
    recorder = RecordingEnv(env)
    p1_kos = p2_kos = 0
    prev_health: tuple[int, int] | None = None
    fight_started = False
    live_p2: list[int] = []
    last_snap = parse_ram(env.unwrapped.get_ram())
    last_match = last_snap.match_counter
    last_p2 = last_snap.p2_character

    def on_frame(_env, frame, snap, _prev) -> bool:
        nonlocal p1_kos, p2_kos, prev_health, fight_started, last_snap
        nonlocal last_match, last_p2
        last_snap = snap
        health = (snap.p1_health, snap.p2_health)
        in_endurance = snap.match_counter in (ENDURANCE1, ENDURANCE1B)
        fight_started = fight_started or (
            snap.match_counter == ENDURANCE1
            and snap.p1_character == LIU_KANG_ID
            and snap.timer > 50
            and health == (161, 161)
        )
        if fight_started and in_endurance:
            if not live_p2 or live_p2[-1] != snap.p2_character:
                live_p2.append(snap.p2_character)
                print(
                    f"  live opponent #{len(live_p2)} "
                    f"id={snap.p2_character}/{char_name(snap.p2_character)} "
                    f"f={frame} {describe(snap)}",
                    flush=True,
                )
        if snap.match_counter != last_match or snap.p2_character != last_p2:
            print(
                f"  change f={frame} match {last_match}->{snap.match_counter} "
                f"p2 {last_p2}/{char_name(last_p2)}->"
                f"{snap.p2_character}/{char_name(snap.p2_character)} "
                f"{describe(snap)}",
                flush=True,
            )
            last_match = snap.match_counter
            last_p2 = snap.p2_character
        if fight_started and in_endurance and prev_health is not None:
            p1_kos += int(prev_health[1] > 0 and health[1] == 0)
            p2_kos += int(prev_health[0] > 0 and health[0] == 0)
        prev_health = health if fight_started and in_endurance else None
        if frame % 500 == 0:
            print(f"  f={frame} kos={p1_kos}-{p2_kos} {describe(snap)}", flush=True)
        if snap.screen is Screen.CONTINUE:
            return True
        return snap.match_counter >= win_at and snap.p1_character == LIU_KANG_ID

    runner = TournamentRunner(
        deterministic=deterministic,
        force_scripted=force_scripted,
        ladder_model=ladder_model,
        on_frame=on_frame,
        policy_loader=make_policy_loader(
            relabel_match,
            kano_script=kano_script,
            courtyard=courtyard,
            round2_kano=round2_kano,
        ),
    )
    apply_oracle(runner, ladder_model=ladder_model, pixel_model=pixel_model)
    result = runner.run_on(recorder, max_frames=max_frames)
    opponents = ",".join(f"{pid}/{char_name(pid)}" for pid in live_p2) or "?"
    won = last_snap.match_counter >= win_at and last_snap.p1_character == LIU_KANG_ID
    print(
        f"  done won={won} frames={len(recorder.masks)} kos={p1_kos}-{p2_kos} "
        f"live_p2={opponents} furthest={result.furthest} {describe(last_snap)}"
    )
    if result.swaps:
        for swap in result.swaps:
            print(f"  swap {swap}")
    return won, recorder.masks, last_snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=40_000)
    parser.add_argument("--max-identify-frames", type=int, default=4_000)
    parser.add_argument("--identify-only", action="store_true")
    parser.add_argument(
        "--win-at",
        type=int,
        default=ENDURANCE2,
        help="match_counter that counts as Endurance 1 cleared (default: 9 = E2).",
    )
    parser.add_argument(
        "--oracles",
        nargs="*",
        default=None,
        help="Subset of oracle labels to try (default: all).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions (tape is still exact RLE once captured).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Attempts per oracle (useful with --stochastic).",
    )
    parser.add_argument(
        "--relabel-match",
        type=int,
        default=None,
        help="Poke match_counter in the oracle's RAM copy only (game is unchanged).",
    )
    parser.add_argument(
        "--round2-kano",
        action="store_true",
        help="After the first Kano KO, switch to scripted keepaway (no jump).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("snes/mortal_kombat/scratch/natural_endurance1_rle.py"),
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("snes/mortal_kombat/recordings/natural_endurance1_start.png"),
    )
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        print("replaying natural fight 1+2+3+4+5+6+7…", flush=True)
        replay_through_fight7(env)
        pin_snap = parse_ram(env.unwrapped.get_ram())
        print(f"pin {describe(pin_snap)}", flush=True)
        if pin_snap.match_counter < ENDURANCE1 or pin_snap.p1_character != LIU_KANG_ID:
            print("fight 7 pin is not at Endurance 1")
            return 1
        pin = get_emulator_state(env.unwrapped)

        ident_frames, ident_snap = identify_live_endurance1(
            env,
            pin,
            max_frames=args.max_identify_frames,
            screenshot=args.screenshot,
        )
        live = ident_snap.match_counter == ENDURANCE1 and is_fight_ready(ident_snap)
        print(
            f"identify live={live} frames={ident_frames} {describe(ident_snap)}",
            flush=True,
        )
        if not live:
            print("Endurance 1 fight never became ready; leftover HUD is not the opponent")
            return 1
        print(
            f"live Endurance 1 opponent is {ident_snap.p2_character}/"
            f"{char_name(ident_snap.p2_character)} "
            "(pin HUD may still show Liu Kang mirror)"
        )
        if ident_snap.p2_character == 4:
            print("live fighter is Scorpion — do not jump into spear")
        if ident_snap.p2_character == 0:
            print("live fighter is Cage — do not jump into shadow kick")
        if ident_snap.p2_character == 5:
            print("live fighter is Sub-Zero — do not jump into ice")
        if args.identify_only:
            return 0

        # Match5 v3 closed M6 det and M7 stoch. Per-stage E1 was 0/34 on
        # save-state; E1B 1/5. Force those zips onto both endurance slots.
        attempts = (
            ("scripted-courtyard", None, True, None),
            ("match5-v3", "mk1_v3_Match5_ppo_final.zip", False, None),
            ("endurance1-v3", "mk1_v3_Endurance1_ppo_final.zip", False, None),
            ("endurance1b-v3", "mk1_v3_Endurance1B_ppo_final.zip", False, None),
            ("fight-v3", "mk1_v3_Fight_ppo_final.zip", False, None),
            ("match2-v3", "mk1_v3_Match2_ppo_final.zip", False, None),
            ("match3-v3", "mk1_v3_Match3_ppo_final.zip", False, None),
            ("match4-v3", "mk1_v3_Match4_ppo_final.zip", False, None),
            ("match6-v3", "mk1_v3_Match6_ppo_final.zip", False, None),
            ("match7-v3", "mk1_v3_Match7_ppo_final.zip", False, None),
            ("per-stage-v3", None, False, None),
            ("scripted", None, True, None),
            ("scripted-kano", None, True, None),
            ("speedrun-pixel", None, False, "mk1_speedrun_ppo_final.zip"),
            ("ladder-ft-pixel", None, False, "mk1_ladder_ft_ppo_final.zip"),
            ("match7-pixel", None, False, "mk1_match7_ppo_9500000_steps.zip"),
            ("ladder-pixel", None, False, "mk1_ladder_ppo_final.zip"),
            ("fresh-pixel", None, False, "mk1_fresh_ppo_final.zip"),
            ("multichar-pixel", None, False, "mk1_multichar_ppo_final.zip"),
        )
        if args.oracles:
            wanted = set(args.oracles)
            attempts = tuple(item for item in attempts if item[0] in wanted)
        for label, ladder, scripted, pixel in attempts:
            for attempt in range(args.repeats):
                tag = label if args.repeats == 1 else f"{label}#{attempt + 1}"
                print(
                    f"oracle {tag} det={not args.stochastic} win_at={args.win_at} "
                    f"relabel={args.relabel_match} round2_kano={args.round2_kano}",
                    flush=True,
                )
                won, masks, snap = capture_from_pin(
                    env,
                    pin,
                    ladder_model=ladder,
                    force_scripted=scripted,
                    max_frames=args.max_frames,
                    pixel_model=pixel,
                    deterministic=not args.stochastic,
                    win_at=args.win_at,
                    relabel_match=args.relabel_match,
                    kano_script=label == "scripted-kano",
                    courtyard=label == "scripted-courtyard",
                    round2_kano=args.round2_kano,
                )
                if not won:
                    continue
                encoded = rle_encode(masks)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                body = (
                    f"NATURAL_ENDURANCE1_FRAMES = {len(masks)}\n"
                    f"{format_rle(encoded)}\n"
                )
                args.out.write_text(body)
                print(f"wrote {args.out} frames={len(masks)} rle={len(encoded)}")
                print(f"end {describe(snap)}")
                return 0
        print(f"no oracle reached match_counter={args.win_at}")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
