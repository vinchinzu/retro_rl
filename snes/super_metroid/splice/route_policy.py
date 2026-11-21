"""Flexible, swappable Route Policy & Skill Registry for Clean speed work.

Lets the user or agent swap any segment or row-range between:
  - ``tape``     — exact verified baseline master-tape buttons (instant rollback)
  - ``skill``    — live controller by registry name or module path
  - ``override`` — searched ``.npz`` button sequence

Local A/B tests cache the chosen tape's natural predecessor states by tape
digest. Full-run policy builds write a new candidate stem and prove it from
power-on via ``replay_tape_chain``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR, SHARED_ROM, VANILLA_ROM_SHA1
from super_metroid.splice.benchmark import FPS_CONVENTION, checkpoint_dir_for, format_rta, save_checkpoint

PROFILES_DIR = GAME_DIR / "policies" / "route_profiles"


@dataclass(frozen=True)
class SkillEntry:
    """One named, reusable Skill callable."""

    skill_id: str
    module: str
    func: str
    description: str
    restore_free: bool = True

    def load(self) -> Callable[[Any], Any]:
        return getattr(importlib.import_module(self.module), self.func)


# Central registry of restore-free RAM-driven Skills available to route policies.
SKILL_REGISTRY: dict[str, SkillEntry] = {
    "ceres.falling_through_ridley": SkillEntry(
        skill_id="ceres.falling_through_ridley",
        module="super_metroid.routes.kpdr.ceres.outbound",
        func="play_ceres_falling_through_ridley",
        description="Falling Tile through Ridley tail tank to the return Magnet Stairs",
    ),
    "ceres.elevator_to_landing": SkillEntry(
        skill_id="ceres.elevator_to_landing",
        module="super_metroid.routes.kpdr.ceres.magnet",
        func="play_ceres_elevator_to_landing",
        description="Ceres right-wall kick, ledge hops and Landing Site handoff",
    ),
    "crateria.bombs_pickup": SkillEntry(
        skill_id="crateria.bombs_pickup",
        module="super_metroid.combat.bomb_torizo",
        func="play_bomb_torizo_pickup",
        description="Bombs pickup from the Flyway entry",
    ),
    "crateria.climb_ascent": SkillEntry(
        skill_id="crateria.climb_ascent",
        module="super_metroid.routes.kpdr.crateria.climb_descent",
        func="play_climb_to_parlor",
        description="Climb from Pit via the platforms and ceiling hatch",
    ),
    "crateria.bomb_torizo": SkillEntry(
        skill_id="crateria.bomb_torizo",
        module="super_metroid.combat.bomb_torizo",
        func="play_bomb_torizo_fast_leave",
        description="Bomb Torizo statue-side fight and Flyway exit",
    ),
    "kraid.warehouse_direct": SkillEntry(
        skill_id="kraid.warehouse_direct",
        module="super_metroid.routes.kpdr.kraid.to_kraid",
        func="play_warehouse_to_zeela_without_hijump",
        description="Warehouse Entrance to Zeela on one Super and a wall jump",
    ),
    "norfair.business_to_frog": SkillEntry(
        skill_id="norfair.business_to_frog",
        module="super_metroid.routes.kpdr.norfair.business_frog",
        func="play_business_to_frog_save",
        description="Business Center elevator -> Frog Savestation direct drop (skips Map Room loop)",
    ),
    "norfair.business_to_ice_gate": SkillEntry(
        skill_id="norfair.business_to_ice_gate",
        module="super_metroid.routes.kpdr.norfair.business_frog",
        func="play_business_to_ice_gate",
        description="Business Center elevator -> Ice Beam Gate Room direct drop (skips 9-row wandering loop)",
    ),
    "norfair.bubble_to_bat_cave": SkillEntry(
        skill_id="norfair.bubble_to_bat_cave",
        module="super_metroid.routes.kpdr.norfair.to_bat_cave",
        func="play_bubble_to_bat_cave_clean",
        description="Bubble Mountain direct climb -> Bat Cave (1-Super Clean, skips double Save Room loop)",
    ),
    "tourian.first_room_direct": SkillEntry(
        skill_id="tourian.first_room_direct",
        module="super_metroid.routes.kpdr.tourian.entry",
        func="play_tourian_first_room_direct",
        description="Tourian First Room elevator -> Metroid Room 1 direct (skips Upper Tourian Save Room)",
    ),
    "tourian.metroid_room_1": SkillEntry(
        skill_id="tourian.metroid_room_1",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_metroid_room_1",
        description="Metroid Room 1 (0xDAE1 -> 0xDB31) restore-free freeze+blast Skill",
    ),
    "tourian.metroid_room_2": SkillEntry(
        skill_id="tourian.metroid_room_2",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_metroid_room_2",
        description="Metroid Room 2 (0xDB31 -> 0xDB7D) restore-free freeze+blast Skill",
    ),
    "tourian.metroid_room_3": SkillEntry(
        skill_id="tourian.metroid_room_3",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_metroid_room_3",
        description="Metroid Room 3 (0xDB7D -> 0xDBCD) restore-free freeze+blast Skill",
    ),
    "tourian.metroid_room_4": SkillEntry(
        skill_id="tourian.metroid_room_4",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_metroid_room_4",
        description="Metroid Room 4 (0xDBCD -> 0xDC19) restore-free freeze+blast Skill",
    ),
    "tourian.metroid_rooms_all": SkillEntry(
        skill_id="tourian.metroid_rooms_all",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_metroid_rooms_all",
        description="Metroid Rooms 1-4 (rows 316-319, 0xDAE1 -> 0xDC19) restore-free Skill",
    ),
    "tourian.big_boy_room": SkillEntry(
        skill_id="tourian.big_boy_room",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_big_boy_room",
        description="Big Boy Room (rows 322-323, 0xDCB1 -> 0xDCFF) restore-free Skill (energy-independent)",
    ),
    "tourian.rinka_shaft_direct": SkillEntry(
        skill_id="tourian.rinka_shaft_direct",
        module="super_metroid.routes.kpdr.tourian.metroid_rooms",
        func="play_rinka_shaft_direct",
        description="Rinka Shaft direct drop -> Mother Brain (rows 328-330, skips Lower Tourian Save Room)",
    ),
    "kraid.varia_to_kraid": SkillEntry(
        skill_id="kraid.varia_to_kraid",
        module="super_metroid.routes.kpdr.kraid.varia_return",
        func="play_varia_to_kraid",
        description="Varia Suit Room -> Kraid Room return",
    ),
    "red_tower.first_ascent": SkillEntry(
        skill_id="red_tower.first_ascent",
        module="super_metroid.routes.kpdr.red_tower.red_stack",
        func="play_first_red_ascent",
        description="Sloaters refill, right-wall kick onto the first frozen Ripper, and climb to Hellway",
    ),
    "red_tower.upper_space_jump": SkillEntry(
        skill_id="red_tower.upper_space_jump",
        module="super_metroid.routes.kpdr.red_tower.red_stack",
        func="play_red_upper_space_jump",
        description="Red Tower upper shaft Space Jumps, ceiling shot and Hellway exit",
    ),
    "wrecked_ship.moat": SkillEntry(
        skill_id="wrecked_ship.moat",
        module="super_metroid.routes.kpdr.wrecked_ship.moat",
        func="play_moat_cross",
        description="Moat shinespark crossing",
    ),
    "wrecked_ship.west_ocean": SkillEntry(
        skill_id="wrecked_ship.west_ocean",
        module="super_metroid.routes.kpdr.wrecked_ship.west_ocean",
        func="play_west_ocean_over_ocean_spark",
        description="West Ocean over-ocean spark to Wrecked Ship",
    ),
    "maridia.space_jump_room_exit": SkillEntry(
        skill_id="maridia.space_jump_room_exit",
        module="super_metroid.routes.kpdr.maridia.post_draygon",
        func="play_space_jump_room_exit",
        description="Space Jump pedestal to the settled Draygon Room entry",
    ),
    "maridia.draygon_room_exit": SkillEntry(
        skill_id="maridia.draygon_room_exit",
        module="super_metroid.routes.kpdr.maridia.post_draygon",
        func="play_draygon_room_exit",
        description="Draygon arena runway and Space Jumps to The Precious Room",
    ),
    "maridia.precious_space_jump": SkillEntry(
        skill_id="maridia.precious_space_jump",
        module="super_metroid.routes.kpdr.maridia.post_draygon",
        func="play_precious_space_jump",
        description="The Precious Room shaft climb to Colosseum",
    ),
    "maridia.halfie_return_drop": SkillEntry(
        skill_id="maridia.halfie_return_drop",
        module="super_metroid.routes.kpdr.maridia.post_draygon",
        func="play_halfie_return_drop",
        description="Halfie Climb direct drop to East Cactus Alley",
    ),
    "maridia.west_cactus_drop": SkillEntry(
        skill_id="maridia.west_cactus_drop",
        module="super_metroid.routes.kpdr.maridia.post_draygon",
        func="play_west_cactus_drop",
        description="West Cactus Alley direct drop to Butterfly Room",
    ),
    "maridia.full_halfie": SkillEntry(
        skill_id="maridia.full_halfie",
        module="super_metroid.routes.kpdr.maridia.halfie_climb",
        func="play_full_halfie_chain",
        description="Botwoon E-Tank -> Halfie Climb -> Colosseum Full Halfie Shinespark (rows 206-208, 0xD7E4 -> 0xD78F)",
    ),
    "maridia.halfie_climb_wj": SkillEntry(
        skill_id="maridia.halfie_climb_wj",
        module="super_metroid.routes.kpdr.maridia.halfie_climb",
        func="play_halfie_climb_walljump",
        description="Halfie Climb Room (0xD913 -> 0xD72A) grappleless 5-kick single-wall-jump climb",
    ),
    "maridia.precious_direct": SkillEntry(
        skill_id="maridia.precious_direct",
        module="super_metroid.routes.kpdr.maridia.halfie_climb",
        func="play_precious_room_direct",
        description="The Precious Room (0xD78F -> 0xDA60) grappleless direct drop to Draygon",
    ),
    "escape.mb3_closeout": SkillEntry(
        skill_id="escape.mb3_closeout",
        module="super_metroid.combat.mother_brain",
        func="play_mother_brain_closeout",
        description="Mother Brain 3 Hyper Beam closeout -> Escape Room 1",
    ),
    "escape.climb": SkillEntry(
        skill_id="escape.climb",
        module="super_metroid.combat.escape",
        func="play_escape_climb",
        description="Crateria Climb escape ascent (dash+walk entry tuning)",
    ),
}


def resolve_skill(target: str) -> tuple[Callable[[Any], Any], str]:
    """Resolve either a registered skill_id or ``module.path:func``."""
    if target in SKILL_REGISTRY:
        entry = SKILL_REGISTRY[target]
        return entry.load(), entry.skill_id
    if ":" in target:
        mod, fn = target.split(":", 1)
        return getattr(importlib.import_module(mod), fn), target
    raise KeyError(f"unknown skill {target!r}; available: {sorted(SKILL_REGISTRY)}")


@dataclass(frozen=True)
class SegmentRule:
    """One swappable segment replacement rule over inclusive board rows."""

    rule_id: str
    start_row: int
    end_row: int
    from_room: str
    to_room: str | None
    mode: str = "tape"           # "tape" | "skill" | "override"
    target: str | None = None    # skill_id or relative .npz path
    enabled: bool = True
    follow_rows: int = 3         # downstream rows checked in local A/B
    name: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SegmentRule:
        mode = str(data.get("mode", "tape"))
        if mode not in ("tape", "skill", "override"):
            raise ValueError(f"invalid rule mode {mode!r} in {data.get('rule_id')!r}")
        return cls(
            rule_id=str(data["rule_id"]),
            start_row=int(data["start_row"]),
            end_row=int(data["end_row"]),
            from_room=str(data["from_room"]),
            to_room=None if data.get("to_room") is None else str(data["to_room"]),
            mode=mode,
            target=None if data.get("target") is None else str(data["target"]),
            enabled=bool(data.get("enabled", True)),
            follow_rows=int(data.get("follow_rows", 3)),
            name=str(data.get("name", data["rule_id"])),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "from_room": self.from_room,
            "to_room": self.to_room,
            "mode": self.mode,
            "target": self.target,
            "enabled": self.enabled,
            "follow_rows": self.follow_rows,
            "notes": self.notes,
        }

    def is_active(self) -> bool:
        return self.enabled and self.mode != "tape"


@dataclass(frozen=True)
class RoutePolicy:
    """Named route policy combining baseline tape with swappable segment rules."""

    policy_id: str
    description: str
    route_variant: str = "early_ice"
    parent: str | None = None
    rules: tuple[SegmentRule, ...] = ()

    def rule_by_id(self, rule_id: str) -> SegmentRule:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(f"rule {rule_id!r} not in policy {self.policy_id!r}")

    def active_rules(self) -> tuple[SegmentRule, ...]:
        active = tuple(sorted((r for r in self.rules if r.is_active()), key=lambda r: r.start_row))
        # Guard against overlapping active row ranges.
        for prev, nxt in zip(active, active[1:]):
            if prev.end_row >= nxt.start_row:
                raise ValueError(
                    f"overlapping active rules {prev.rule_id!r} ({prev.start_row}..{prev.end_row}) "
                    f"and {nxt.rule_id!r} ({nxt.start_row}..{nxt.end_row})"
                )
        return active

    def with_toggles(
        self,
        *,
        enable: Sequence[str] = (),
        disable: Sequence[str] = (),
        use_tape: Sequence[str] = (),
    ) -> RoutePolicy:
        """Return a new RoutePolicy with CLI toggles applied (instant swap/rollback)."""
        known = {r.rule_id for r in self.rules}
        for rid in (*enable, *disable, *use_tape):
            if rid not in known:
                raise KeyError(f"unknown rule {rid!r} in policy {self.policy_id!r}")
        en_set, dis_set, tape_set = set(enable), set(disable), set(use_tape)
        updated: list[SegmentRule] = []
        for r in self.rules:
            nr = r
            if r.rule_id in en_set:
                nr = replace(nr, enabled=True)
            if r.rule_id in dis_set:
                nr = replace(nr, enabled=False)
            if r.rule_id in tape_set:
                nr = replace(nr, mode="tape", enabled=False)
            updated.append(nr)
        return replace(self, rules=tuple(updated))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "description": self.description,
            "route_variant": self.route_variant,
            "parent": self.parent,
            "rules": [r.to_dict() for r in self.rules],
        }


def list_policy_ids(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    if not profiles_dir.is_dir():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.json"))


def load_policy(policy_ref: str | Path, *, profiles_dir: Path = PROFILES_DIR) -> RoutePolicy:
    """Load a RoutePolicy by ID or path, resolving single-parent inheritance."""
    path = Path(policy_ref)
    if not path.is_file():
        path = profiles_dir / f"{policy_ref}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"route policy {policy_ref!r} not found (checked {path}); "
            f"available: {list_policy_ids(profiles_dir)}"
        )
    raw = json.loads(path.read_text())
    parent_id = raw.get("parent")
    merged_rules: dict[str, SegmentRule] = {}
    if parent_id:
        parent = load_policy(parent_id, profiles_dir=profiles_dir)
        for r in parent.rules:
            merged_rules[r.rule_id] = r
    for r_raw in raw.get("rules", ()):
        rule = SegmentRule.from_dict(r_raw)
        merged_rules[rule.rule_id] = rule
    ordered = tuple(sorted(merged_rules.values(), key=lambda r: (r.start_row, r.rule_id)))
    return RoutePolicy(
        policy_id=str(raw.get("policy_id", path.stem)),
        description=str(raw.get("description", "")),
        route_variant=str(raw.get("route_variant", "early_ice")),
        parent=parent_id,
        rules=ordered,
    )


def _row_slices(source_stem: Path) -> tuple[dict[str, Any], Any, dict[int, tuple[int, int, dict[str, Any]]]]:
    """Return (report, actions_array, {row_num: (offset, length, row_dict)})."""
    import numpy as np

    report = json.loads(source_stem.with_suffix(".json").read_text())
    actions = np.load(source_stem.with_suffix(".npz"))["actions"]
    if hashlib.sha256(actions.tobytes()).hexdigest() != report["tape_sha256"]:
        raise ValueError(f"tape digest mismatch for {source_stem}")
    slices: dict[int, tuple[int, int, dict[str, Any]]] = {}
    offset = 0
    for r in report["rows"]:
        n = int(r["frames"])
        slices[int(r["row"])] = (offset, n, r)
        offset += n
    return report, actions, slices


def _policy_predecessor(rule: SegmentRule, slices: Mapping[int, tuple[int, int, dict[str, Any]]]) -> int:
    if rule.start_row not in slices:
        raise ValueError(f"{rule.rule_id}: row {rule.start_row} is absent from source tape")
    end = max(n for n in slices if n <= rule.end_row)
    meta = slices[end][2]
    if int(meta.get("replaced_rows", (end, end))[1]) != rule.end_row:
        raise ValueError(f"{rule.rule_id}: row {rule.end_row} is not a source tape boundary")
    return max((n for n in slices if n < rule.start_row), default=0)


def _step_actions_until_goal(
    env: Any,
    body: Any,
    target_room: int,
    *,
    continuation: bool = False,
    goal_phase: str | None = None,
    max_settle: int = 400,
) -> Any | None:
    """Play binary SNES-12 actions; return full executed array on settled reach."""
    import numpy as np
    from super_metroid.ram import GameplayPhase, parse_env_state

    start_room = parse_env_state(env, mode="nav").room_id
    early_stop = (not continuation) and (goal_phase is None) and (start_room != target_room)

    def reached(s: Any) -> bool:
        if goal_phase == "ending_or_credits":
            return s.phase.value == "ending_or_credits" and s.health > 0
        return s.room_id == target_room and s.game_state == 8 and not s.door_transition and s.health > 0

    executed = []
    st = parse_env_state(env, mode="nav")
    for action in body:
        env.step(action)
        executed.append(action)
        st = parse_env_state(env, mode="nav")
        if st.phase is GameplayPhase.DEATH_OR_GAME_OVER or (st.game_state == 8 and st.health <= 0):
            return None
        if early_stop and reached(st):
            return np.asarray(executed, dtype=np.int8)

    if reached(st):
        return np.asarray(executed, dtype=np.int8)
    if continuation:
        return None
    idle = np.zeros(12, dtype=np.int8)
    for _ in range(max_settle):
        action = idle if st.room_id == target_room else body[-1]
        env.step(action)
        executed.append(action)
        st = parse_env_state(env, mode="nav")
        if st.phase is GameplayPhase.DEATH_OR_GAME_OVER or st.health <= 0:
            return None
        if reached(st):
            return np.asarray(executed, dtype=np.int8)
    return None


def _retime_saved_row(
    env: Any,
    saved_body: Any,
    row_meta: Mapping[str, Any],
    *,
    max_delay: int = 24,
) -> tuple[Any | None, str, int]:
    """Try saved buttons verbatim, then small entry idle delays or trims."""
    import numpy as np

    target = int(row_meta.get("to_room") or row_meta["room"], 16)
    cont = bool(row_meta.get("continuation"))
    gphase = row_meta.get("goal_phase")
    snap = env.em.get_state()
    restores = 0
    idle = np.zeros((1, 12), dtype=np.int8)

    try:
        for delay in range(max_delay + 1):
            env.em.set_state(snap)
            restores += 1
            cand = saved_body if delay == 0 else np.concatenate((np.repeat(idle, delay, axis=0), saved_body))
            found = _step_actions_until_goal(env, cand, target, continuation=cont, goal_phase=gphase)
            if found is not None:
                return found, f"saved+idle:{delay}", restores
        for trim in range(1, min(max_delay + 1, len(saved_body))):
            env.em.set_state(snap)
            restores += 1
            found = _step_actions_until_goal(env, saved_body[trim:], target, continuation=cont, goal_phase=gphase)
            if found is not None:
                return found, f"saved-trim:{trim}", restores
        return None, "failed", restores
    finally:
        env.em.set_state(snap)


def _rejoin_row(
    env: Any, body: Any, row: Mapping[str, Any], *, max_delay: int,
) -> tuple[Any | None, str, int]:
    from super_metroid.ram import parse_env_state
    from super_metroid.splice.assemble import _chain_controller, _record_chain_controller

    skill = row.get("controller_id") if row.get("policy_rule_id") else None
    restores = 0
    if not skill:
        found, method, restores = _retime_saved_row(env, body, row, max_delay=max_delay)
        if found is not None:
            return found, method, restores
    snap = env.em.get_state()
    try:
        if skill:
            play, _ = resolve_skill(skill)
            result = _record_chain_controller(env, play)
        else:
            result = _chain_controller(env, row)
        if result is not None:
            leave = parse_env_state(env, mode="nav")
            target = int(row.get("to_room") or row["room"], 16)
            if leave.room_id != target or leave.game_state != 8 or leave.door_transition or leave.health <= 0:
                result = None
    except (RuntimeError, TimeoutError):
        result = None
    finally:
        env.em.set_state(snap)
        restores += 1
    if result is not None:
        return result[0], "room-controller", restores + result[1]
    if skill:
        found, method, extra = _retime_saved_row(env, body, row, max_delay=max_delay)
        return found, method, restores + extra
    return None, "failed", restores


def _execute_rule(
    env: Any,
    rule: SegmentRule,
    source_actions: Any,
    slices: Mapping[int, tuple[int, int, dict[str, Any]]],
) -> tuple[Any, str, int]:
    """Run one rule (tape, skill, or override) from the current live state."""
    import numpy as np
    from super_metroid.splice.assemble import _record_chain_controller

    if rule.mode == "tape":
        chunks = []
        for rnum in range(rule.start_row, rule.end_row + 1):
            if rnum in slices:
                off, n, _ = slices[rnum]
                chunks.append(source_actions[off : off + n])
        body = np.concatenate(chunks, axis=0)
        for act in body:
            env.step(act)
        return body, "tape", 0

    if rule.mode == "skill":
        if not rule.target:
            raise ValueError(f"skill rule {rule.rule_id!r} missing target")
        play_fn, skill_id = resolve_skill(rule.target)
        actions, restores = _record_chain_controller(env, play_fn)
        return actions, f"skill:{skill_id}", restores

    if rule.mode == "override":
        if not rule.target:
            raise ValueError(f"override rule {rule.rule_id!r} missing target")
        p = Path(rule.target)
        if not p.is_absolute():
            p = GAME_DIR / p
        body = np.load(p)["actions"]
        target = int(rule.to_room or rule.from_room, 16)
        found = _step_actions_until_goal(env, body, target)
        if found is None:
            raise RuntimeError(f"override {p} did not reach {rule.to_room or rule.from_room}")
        return found, f"override:{p.name}", 0

    raise ValueError(f"unsupported rule mode {rule.mode!r}")


def _state_summary(env: Any) -> dict[str, Any]:
    from super_metroid.ram import parse_env_state

    st = parse_env_state(env, mode="full")
    return {
        "room": f"0x{st.room_id:04X}",
        "xy": [st.samus_x, st.samus_y],
        "pose": st.pose,
        "game_state": st.game_state,
        "door_transition": st.door_transition,
        "energy": st.health,
        "max_energy": st.max_health,
        "reserve": st.reserve_health,
        "missiles": st.missiles,
        "max_missiles": st.max_missiles,
        "supers": st.super_missiles,
        "max_supers": st.max_super_missiles,
        "power_bombs": st.power_bombs,
        "max_power_bombs": st.max_power_bombs,
        "selected_item": st.selected_item,
        "items": f"0x{st.collected_items:04X}",
        "beams": f"0x{st.collected_beams:04X}",
    }


def ab_test_policy_rule(
    policy: RoutePolicy,
    rule_id: str,
    *,
    source_stem: Path,
    checkpoint_dir: Path | None = None,
    follow_rows: int | None = None,
    max_delay: int = 24,
    save_npz: Path | None = None,
) -> dict[str, Any]:
    """Run a local A/B test comparing a rule's candidate vs baseline tape.

    Reconstructs the selected tape's natural entry on a cache miss and uses
    that entry to compare the candidate and its downstream join with source.
    """
    import numpy as np
    from retro_harness.env import read_state_bytes
    from super_metroid.dev.common import make_dev_env
    from super_metroid.splice.assemble import _guard_writes

    rule = policy.rule_by_id(rule_id)
    n_follow = rule.follow_rows if follow_rows is None else follow_rows
    report, source_actions, slices = _row_slices(source_stem)
    predecessor = _policy_predecessor(rule, slices)
    ckpt_dir = checkpoint_dir or checkpoint_dir_for(report["tape_sha256"])
    ckpt = ckpt_dir / f"row{predecessor:03d}_leave.state"
    all_row_nums = [int(r["row"]) for r in report["rows"]]
    downstream_nums = [n for n in all_row_nums if n > rule.end_row][:n_follow]

    env = make_dev_env()
    env.reset()
    restore_writes = _guard_writes(env)
    try:
        if ckpt.is_file():
            ckpt_bytes = read_state_bytes(ckpt)
            env.em.set_state(ckpt_bytes)
        else:
            for action in source_actions[:slices[rule.start_row][0]]:
                env.step(action)
            ckpt_bytes = env.em.get_state()
            save_checkpoint(ckpt, ckpt_bytes)

        # 1. Baseline (old tape code)
        entry_summary = _state_summary(env)
        if int(rule.from_room, 16) and entry_summary["room"] != rule.from_room:
            raise ValueError(f"{rule.rule_id}: entered {entry_summary['room']}, expected {rule.from_room}")
        base_seg_frames = 0
        for rnum in range(rule.start_row, rule.end_row + 1):
            if rnum in slices:
                off, n, _ = slices[rnum]
                for act in source_actions[off : off + n]:
                    env.step(act)
                base_seg_frames += n
        base_leave = _state_summary(env)
        base_follow_frames = 0
        for rnum in downstream_nums:
            off, n, _ = slices[rnum]
            for act in source_actions[off : off + n]:
                env.step(act)
            base_follow_frames += n
        base_end = _state_summary(env)

        # 2. Candidate (new skill / override)
        env.em.set_state(ckpt_bytes)
        cand_error: str | None = None
        cand_method = "failed"
        cand_restores = 0
        cand_seg_frames = 0
        cand_actions: Any = None
        cand_leave: dict[str, Any] | None = None
        cand_follow_frames = 0
        downstream_results: list[dict[str, Any]] = []
        downstream_ok = False
        cand_end: dict[str, Any] | None = None

        try:
            cand_actions, cand_method, cand_restores = _execute_rule(env, rule, source_actions, slices)
            cand_seg_frames = len(cand_actions)
            cand_leave = _state_summary(env)
            want_room = rule.to_room or rule.from_room
            if cand_leave["room"] != want_room or cand_leave["game_state"] != 8 or cand_leave["door_transition"] != 0:
                raise RuntimeError(
                    f"candidate left in {cand_leave['room']} gs={cand_leave['game_state']} "
                    f"door={cand_leave['door_transition']}, expected settled {want_room}"
                )
            if save_npz is not None:
                save_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_npz, actions=np.asarray(cand_actions, dtype=np.int8))

            # Verify downstream Join / Sync across follow_rows.
            downstream_ok = True
            for rnum in downstream_nums:
                off, n, meta = slices[rnum]
                saved_body = source_actions[off : off + n]
                found, method, r_cnt = _rejoin_row(env, saved_body, meta, max_delay=max_delay)
                cand_restores += r_cnt
                if found is None:
                    downstream_ok = False
                    downstream_results.append({
                        "row": rnum, "name": meta["name"], "ok": False, "method": "failed",
                        "baseline_frames": n, "candidate_frames": None,
                    })
                    break
                for act in found:
                    env.step(act)
                cand_follow_frames += len(found)
                downstream_results.append({
                    "row": rnum, "name": meta["name"], "ok": True, "method": method,
                    "baseline_frames": n, "candidate_frames": len(found),
                    "delta_frames": n - len(found),
                })
            cand_end = _state_summary(env)
        except Exception as exc:
            cand_error = f"{type(exc).__name__}: {exc}"
    finally:
        restore_writes()
        env.close()

    base_total = base_seg_frames + base_follow_frames
    cand_total = cand_seg_frames + cand_follow_frames
    local_saved = (base_seg_frames - cand_seg_frames) if cand_error is None else None
    net_saved = (base_total - cand_total) if (cand_error is None and downstream_ok) else None
    verdict = "PASS" if (cand_error is None and downstream_ok) else "FAIL"

    return {
        "schema_version": 1,
        "kind": "policy_rule_ab",
        "policy_id": policy.policy_id,
        "rule": rule.to_dict(),
        "checkpoint": str(ckpt),
        "entry": entry_summary,
        "verdict": verdict,
        "error": cand_error,
        "candidate_method": cand_method,
        "rollout_restores": cand_restores,
        "baseline": {
            "segment_frames": base_seg_frames,
            "follow_frames": base_follow_frames,
            "total_frames": base_total,
            "leave": base_leave,
            "end": base_end,
        },
        "candidate": {
            "segment_frames": cand_seg_frames,
            "follow_frames": cand_follow_frames,
            "total_frames": cand_total,
            "leave": cand_leave,
            "end": cand_end,
        },
        "local_frames_saved": local_saved,
        "local_seconds_saved": None if local_saved is None else round(local_saved / FPS_CONVENTION, 2),
        "net_frames_saved": net_saved,
        "net_seconds_saved": None if net_saved is None else round(net_saved / FPS_CONVENTION, 2),
        "energy_delta_at_end": (
            None if (cand_end is None) else cand_end["energy"] - base_end["energy"]
        ),
        "downstream_results": downstream_results,
    }


def build_policy_chain(
    policy: RoutePolicy,
    *,
    source_stem: Path,
    out_stem: Path,
    prefix_stem: Path | None = None,
    max_delay: int = 24,
    checkpoint_dir: Path | None = None,
    use_checkpoint_seed: bool = True,
    verify_power_on: bool = True,
    through_row: int | None = None,
) -> dict[str, Any]:
    """Materialize a full power-on Clean candidate master tape from a RoutePolicy.

    Untouched rows copy verbatim buttons from ``source_stem``; active rules run
    their live Skills/overrides and collapse bypassed detour rows cleanly.
    An optional ``prefix_stem`` supplies the rows before the first active rule.
    When ``use_checkpoint_seed`` is True and the source's predecessor checkpoint
    exists, the development build phase copies the verified prefix buttons and
    builds the suffix directly from that checkpoint (~10x faster iteration);
    ``verify_power_on=True`` then proves the entire combined tape from
    ``retro.State.NONE`` with zero loads, zero RAM writes, and zero deaths.
    """
    import numpy as np
    from retro_harness.env import read_state_bytes, write_state_bytes
    from super_metroid.combat.escape import finish_controller_for, play_credits_tail
    from super_metroid.dev.common import make_dev_env
    from super_metroid.ram import credits_complete, parse_env_state
    from super_metroid.routes.tips import play_hops
    from super_metroid.splice.assemble import (
        AssemblyHop,
        _chain_session,
        _guard_writes,
        _record_chain_controller,
        replay_tape_chain,
    )

    if not out_stem.name.endswith("_clean"):
        raise ValueError("policy build requires a _clean output stem")
    if out_stem.resolve() == source_stem.resolve():
        raise ValueError("out_stem must differ from source_stem (never overwrite baseline directly)")
    if prefix_stem is not None and out_stem.resolve() == prefix_stem.resolve():
        raise ValueError("out_stem must differ from prefix_stem")

    rom_bytes = SHARED_ROM.read_bytes()
    rom_sha1 = hashlib.sha1(rom_bytes).hexdigest()
    if rom_sha1 != VANILLA_ROM_SHA1:
        raise ValueError(f"vanilla ROM required ({VANILLA_ROM_SHA1}), got {rom_sha1}")

    source_report, source_actions, slices = _row_slices(source_stem)
    active_rules = policy.active_rules()
    if through_row is not None:
        boundaries = {meta.get("replaced_rows", (n, n))[1] for n, (_, _, meta) in slices.items()}
        if through_row not in boundaries:
            raise ValueError(f"row {through_row} is not a source tape boundary")
        if any(rule.start_row <= through_row < rule.end_row for rule in active_rules):
            raise ValueError(f"row {through_row} cuts through an active policy rule")
        active_rules = [rule for rule in active_rules if rule.end_row <= through_row]
    predecessors = {rule.start_row: _policy_predecessor(rule, slices) for rule in active_rules}
    active_by_start = {rule.start_row: rule for rule in active_rules}
    first_active = min(active_by_start, default=None)
    prefix_report, prefix_actions, prefix_slices = source_report, source_actions, slices
    if prefix_stem is not None:
        if first_active is None or predecessors[first_active] == 0:
            raise ValueError("a prefix stem requires a first active rule after the opening row")
        prefix_report, prefix_actions, prefix_slices = _row_slices(prefix_stem)
    prefix_rows = [
        row for row in prefix_report["rows"]
        if first_active is not None and int(row["row"]) < first_active
    ]
    prefix_end = int(prefix_rows[-1]["row"]) if prefix_rows else 0
    if prefix_stem is not None:
        source_end = predecessors[first_active]
        source_meta = slices[source_end][2]
        prefix_meta = prefix_rows[-1] if prefix_rows else {}
        if (
            prefix_meta.get("replaced_rows", (prefix_end, prefix_end))[-1]
            != source_meta.get("replaced_rows", (source_end, source_end))[-1]
            or (prefix_meta.get("to_room") or prefix_meta.get("room"))
            != (source_meta.get("to_room") or source_meta.get("room"))
        ):
            raise ValueError("prefix tape must reach the source boundary before the first active rule")
    skipped_rows: set[int] = set()
    for r in active_rules:
        for n in range(r.start_row + 1, r.end_row + 1):
            skipped_rows.add(n)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_stem.with_suffix(".json")
    tape_path = out_stem.with_suffix(".npz")
    if report_path.exists() or tape_path.exists():
        raise FileExistsError(f"{out_stem} already exists. Choose a new --stem for this build.")

    ledger: list[dict[str, Any]] = []
    master: list[Any] = []
    rule_summaries: list[dict[str, Any]] = []
    rollout_restores = 0
    started = time.monotonic()

    env = make_dev_env()
    env.reset()
    restore_writes = _guard_writes(env)
    session = _chain_session(env)
    splits: list[Any] = []

    def commit(actions_arr: Any, label: str, target: int) -> None:
        def play(live: Any) -> None:
            for act in actions_arr:
                live.step(act, label)

        play_hops(session, splits, [AssemblyHop(label, play, session.state.room_id, target, label, label)])
        master.extend(actions_arr)

    diverged_from_source = False
    stopped: dict[str, Any] | None = None

    ckpt_dir = checkpoint_dir or checkpoint_dir_for(prefix_report["tape_sha256"])
    seed_ckpt = (
        ckpt_dir / f"row{prefix_end:03d}_leave.state"
        if use_checkpoint_seed and prefix_end
        else None
    )
    seeded_prefix = seed_ckpt is not None and seed_ckpt.is_file()

    try:
        for prefix_row in prefix_rows:
            rnum = int(prefix_row["row"])
            off, n, _ = prefix_slices[rnum]
            body = prefix_actions[off : off + n]
            if seeded_prefix:
                master.extend(body)
            else:
                target = int(prefix_row.get("to_room") or prefix_row["room"], 16)
                commit(body, f"policy:row:{rnum}", target)
            ledger.append(dict(prefix_row))
        if seeded_prefix:
            env.em.set_state(read_state_bytes(seed_ckpt))
            session.frame = len(master)
            session.state = parse_env_state(env, frame=session.frame, mode="nav")
            print(
                f"[policy] seeded prefix through row {prefix_end} ({len(master):,}f) "
                f"from {seed_ckpt.name}; building suffix...",
                flush=True,
            )

        for src_row in source_report["rows"]:
            rnum = int(src_row["row"])
            if through_row is not None and rnum > through_row:
                break
            if first_active is not None and rnum < first_active:
                continue
            if rnum in skipped_rows:
                continue

            if rnum in active_by_start:
                rule = active_by_start[rnum]
                base_frames = sum(slices[n][1] for n in range(rule.start_row, rule.end_row + 1) if n in slices)
                if int(rule.from_room, 16) not in (0, session.state.room_id):
                    stopped = {"row": rnum, "rule_id": rule.rule_id, "error": f"expected entry {rule.from_room}"}
                    break
                snap = env.em.get_state()
                try:
                    cand_actions, cand_method, r_cnt = _execute_rule(env, rule, source_actions, slices)
                    rollout_restores += r_cnt
                except (RuntimeError, TimeoutError) as exc:
                    stopped = {"row": rnum, "rule_id": rule.rule_id, "error": str(exc)}
                    break
                finally:
                    env.em.set_state(snap)
                    rollout_restores += 1
                target = int(rule.to_room or rule.from_room, 16)
                commit(cand_actions, f"policy:{rule.rule_id}", target)
                st = session.state
                if st.room_id != target or st.game_state != 8 or st.door_transition != 0 or st.health <= 0:
                    stopped = {
                        "row": rnum, "rule_id": rule.rule_id,
                        "error": f"skill left in 0x{st.room_id:04X} gs={st.game_state} door={st.door_transition} E={st.health}",
                    }
                    break
                end_meta = slices[rule.end_row][2] if rule.end_row in slices else src_row
                ledger.append({
                    "row": rnum,
                    "segment": src_row.get("segment"),
                    "hop_index": src_row.get("hop_index"),
                    "room": rule.from_room,
                    "to_room": rule.to_room,
                    "name": rule.name or src_row.get("name"),
                    "tape": src_row.get("tape"),
                    "source_start": src_row.get("source_start"),
                    "source_end": end_meta.get("source_end"),
                    "controller_id": rule.target if rule.mode == "skill" else None,
                    "continuation": bool(end_meta.get("continuation", False)),
                    "goal_phase": end_meta.get("goal_phase"),
                    "source_sha256": src_row.get("source_sha256"),
                    "frames": len(cand_actions),
                    "method": "room-controller" if rule.mode == "skill" else cand_method,
                    "policy_rule_id": rule.rule_id,
                    "replaced_rows": [rule.start_row, rule.end_row],
                    "energy": st.health,
                    "max_energy": st.max_health,
                    "leave_x": st.samus_x,
                    "leave_y": st.samus_y,
                })
                saved = base_frames - len(cand_actions)
                rule_summaries.append({
                    "rule_id": rule.rule_id,
                    "rows": [rule.start_row, rule.end_row],
                    "baseline_frames": base_frames,
                    "candidate_frames": len(cand_actions),
                    "frames_saved": saved,
                    "seconds_saved": round(saved / FPS_CONVENTION, 2),
                    "method": cand_method,
                    "energy_after": st.health,
                })
                diverged_from_source = True
                print(
                    f"[policy] {rule.rule_id} (rows {rule.start_row}..{rule.end_row}): "
                    f"{len(cand_actions)}f vs {base_frames}f (saved {saved:+d}f / {saved/FPS_CONVENTION:+.2f}s), "
                    f"E={st.health}/{st.max_health}",
                    flush=True,
                )
                continue

            # Standard row: use verified source buttons verbatim if synchronized,
            # or retime / finish_controller if upstream policy shifted state.
            off, n, _ = slices[rnum]
            saved_body = source_actions[off : off + n]
            target = int(src_row.get("to_room") or src_row["room"], 16)
            method = str(src_row.get("method", "tape+idle:0"))
            chosen = saved_body

            if diverged_from_source:
                # Try live finish controller first for post-Rainbow rows (332-339).
                fin_ctrl = finish_controller_for(
                    session.state, src_row["room"], src_row.get("to_room"), src_row.get("goal_phase"),
                )
                if fin_ctrl is not None:
                    snap = env.em.get_state()
                    try:
                        chosen, r_cnt = _record_chain_controller(env, fin_ctrl)
                        rollout_restores += r_cnt
                        method = "room-controller"
                    finally:
                        env.em.set_state(snap)
                        rollout_restores += 1
                else:
                    found, retime_method, r_cnt = _rejoin_row(env, saved_body, src_row, max_delay=max_delay)
                    rollout_restores += r_cnt
                    if found is None:
                        stopped = {
                            "row": rnum,
                            "room": src_row["room"],
                            "name": src_row.get("name"),
                            "error": f"downstream row {rnum} ({src_row.get('name')}) failed to re-sync after policy change",
                        }
                        print(f"[policy] STOP at row {rnum}: {stopped['error']}", flush=True)
                        break
                    chosen = found
                    if retime_method != "saved+idle:0" or src_row.get("policy_rule_id"):
                        method = retime_method

            commit(chosen, f"policy:row:{rnum}", target)
            st = session.state
            ledger.append(
                dict(src_row) | {
                    "frames": len(chosen),
                    "method": method,
                    "energy": st.health,
                    "max_energy": st.max_health,
                    "leave_x": st.samus_x,
                    "leave_y": st.samus_y,
                }
            )

        credits_tail_frames = 0
        credits_done = False
        if stopped is None and ledger and ledger[-1].get("goal_phase") == "ending_or_credits":
            if not credits_complete(env):
                snap = env.em.get_state()
                try:
                    tail = play_credits_tail(env)
                finally:
                    env.em.set_state(snap)
                    rollout_restores += 1
                commit(tail, "policy:credits", session.state.room_id)
                credits_tail_frames = len(tail)
            credits_done = bool(credits_complete(env))

        actions_arr = np.asarray(master, dtype=np.int8).reshape((-1, 12))
        np.savez_compressed(tape_path, actions=actions_arr)
        out_report: dict[str, Any] = {
            "kind": "clean_tape_chain",
            "schema_version": 1,
            "board": source_report.get("board"),
            "policy_id": policy.policy_id,
            "source_stem": str(source_stem),
            **({"prefix_stem": str(prefix_stem)} if prefix_stem is not None else {}),
            "policy_rules_applied": rule_summaries,
            "rows": ledger,
            "rollout_restores": rollout_restores,
            "stopped": stopped,
            "credits_complete": credits_done,
            "credits_tail_frames": credits_tail_frames,
            "rom_sha1": rom_sha1,
            "rom_sha256": hashlib.sha256(rom_bytes).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_frames": session.frame,
            "rooms_cleared": sum(bool(r.get("to_room")) and r["room"] != "0x0000" for r in ledger),
            "final_state": parse_env_state(env, frame=session.frame).to_dict(),
            "state_parse": "full_bank7e",
            "seconds": round(time.monotonic() - started, 2),
            "tape_sha256": hashlib.sha256(actions_arr.tobytes()).hexdigest(),
            "endpoint_sha256": hashlib.sha256(env.em.get_state()).hexdigest(),
        }
        report_path.write_text(json.dumps(out_report, indent=2) + "\n")
        write_state_bytes(SCRATCH_STATE_DIR / f"{out_stem.name}.state", env.em.get_state())
    finally:
        restore_writes()
        env.close()

    # Independent loadless, write-free power-on proof (always required for acceptance)!
    if verify_power_on and stopped is None:
        proof = replay_tape_chain(stem=out_stem)
        if proof["success"]:
            out_report["build_endpoint_sha256"] = out_report["endpoint_sha256"]
            out_report["endpoint_sha256"] = proof["core_snapshot_sha256"]
        out_report["proof"] = proof
        out_report["credits_complete"] = bool(proof["credits_complete"])
        report_path.write_text(json.dumps(out_report, indent=2) + "\n")
    return out_report


def format_policy_list(profiles_dir: Path = PROFILES_DIR) -> str:
    lines = ["# Route Policies & Registered Skills", "", "## Route Profiles", ""]
    for pid in list_policy_ids(profiles_dir):
        pol = load_policy(pid, profiles_dir=profiles_dir)
        active = [r.rule_id for r in pol.active_rules()]
        lines.append(
            f"- **`{pol.policy_id}`** (`{pol.route_variant}`): {pol.description} "
            f"| active rules: {', '.join(f'`{a}`' for a in active) if active else 'none (100% baseline tape)'}"
        )
    lines.extend(["", "## Registered Restore-Free Skills", ""])
    for sid, entry in sorted(SKILL_REGISTRY.items()):
        lines.append(f"- **`{sid}`** (`{entry.module}:{entry.func}`): {entry.description}")
    return "\n".join(lines) + "\n"


def format_ab_report(ab: Mapping[str, Any]) -> str:
    rule = ab["rule"]
    base = ab["baseline"]
    cand = ab["candidate"]
    lines = [
        f"## A/B Result: `{rule['rule_id']}` ({ab['verdict']})",
        f"- **Rows:** {rule['start_row']}..{rule['end_row']} ({rule['from_room']} → {rule['to_room']}) | **Mode:** `{rule['mode']}` (`{rule['target']}`)",
        f"- **Checkpoint:** `{Path(ab['checkpoint']).name}` (entry E={ab['entry']['energy']}/{ab['entry']['max_energy']}, M={ab['entry']['missiles']}, S={ab['entry']['supers']}, PB={ab['entry']['power_bombs']})",
        f"- **Local segment:** {cand['segment_frames']:,}f vs {base['segment_frames']:,}f "
        f"(saved {ab['local_frames_saved']}f / {format_rta(ab['local_seconds_saved'])})" if ab["local_frames_saved"] is not None else f"- **Error:** {ab['error']}",
    ]
    if ab["net_frames_saved"] is not None:
        lines.append(
            f"- **Net through +{len(ab['downstream_results'])} follow rows:** "
            f"{cand['total_frames']:,}f vs {base['total_frames']:,}f "
            f"(**saved {ab['net_frames_saved']:+,}f / {ab['net_seconds_saved']:+.2f}s**, "
            f"energy delta {ab['energy_delta_at_end']:+d})"
        )
    for dr in ab["downstream_results"]:
        status = "OK" if dr["ok"] else "FAIL"
        lines.append(f"  - row {dr['row']} ({dr['name']}): {status} via `{dr['method']}` ({dr['candidate_frames']}f vs {dr['baseline_frames']}f)")
    return "\n".join(lines) + "\n"
