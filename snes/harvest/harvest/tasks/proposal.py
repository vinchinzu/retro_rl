"""Native courting and Sunday feather purchase for the day composer."""

from dataclasses import dataclass, field, replace
from collections import deque

from retro_harness import ActionResult, Task, TaskResult, TaskStatus
from harvest.core.carry import tool_in_carry_pair
from harvest.core.ram_catalog import read_ram_u8, read_ram_u16, read_ram_value
from harvest.core.npc_catalog import current_dialogue_registers, game_objects, text_record_for_id
from harvest.core.tile_catalog import is_path_tilemap
from harvest.maps.map_routes import ROUTES, segment_waypoints
from harvest.maps.map_types import Waypoint
from harvest.planner.tasks.multi_nav import MultiMapNavTask
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.inventory_exit import room_walk_action
from harvest.planner.world_probe import WorldProbe
from harvest.tasks.mountain_berry import MountainBerryTask
from harvest.tasks.nav import get_pos_from_ram, get_tile_at, make_action
from harvest.tasks.primitives import drain_action_queue, press_a_sequence, press_button_sequence


def marriage_phases_for_ram(ram):
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    probe = WorldProbe(ram=ram)
    if read_ram_value(ram, "marriage_flags") & 31 or probe.is_storm() or probe.is_festival():
        return []
    hearts = read_ram_value(ram, "ann_hearts")
    owned = tool_in_carry_pair(ram, 0x18) or read_ram_u8(ram, 0x11F02) & 0x80
    if (probe.weekday() == 0 and probe.clock().hour < 16 and not owned
            and read_ram_value(ram, "money") >= 1000
            and (hearts >= 250 or read_ram_value(ram, "upgrade_flags") & 0x40)
            and not read_ram_value(ram, "family_event_flags") & 0x40):
        return [PhaseSpec("BUY_FEATHER", PhaseKind.FEATHER_BUY, failure_policy="optional")]
    opening = (read_ram_value(ram, "year") == 0 and read_ram_value(ram, "season") == 0
               and read_ram_value(ram, "day") <= 2)
    if hearts < 250 and not opening and probe.weekday() != 0 and probe.clock().hour < 10:
        return [PhaseSpec("COURT_ANN", PhaseKind.COURT, failure_policy="optional")]
    return []


# Bar talks stop at 108 so the Flower Festival's +42 clears the 120–149 dead zone.
# Summer and Fall pay nothing; the night task is what `bar_talk_outlook` allows.
_EVE_HOLD = 108
_EVE_FLOOR = 250
_FEATHER = 0x18
_PROPOSE_MAPS = frozenset({0x00, 0x01, 0x02, 0x03, 0x0C, 0x10})


def _eve_married(ram) -> bool:
    from harvest.tasks.eve_proposal import ADDR_WIFE_FLAGS

    return bool(read_ram_u16(ram, ADDR_WIFE_FLAGS) & 0x1F)


def _eve_outlook(ram, probe):
    from harvest.tasks.eve_loop_task import ADDR_WEATHER_FLAGS, bar_talk_outlook

    season, day = probe.calendar_date()
    return bar_talk_outlook(
        read_ram_value(ram, "eve_hearts"),
        season=season,
        day=day,
        weather_flags=read_ram_u16(ram, ADDR_WEATHER_FLAGS, live_offset=False),
    )


def _feather_owned(ram) -> bool:
    return tool_in_carry_pair(ram, _FEATHER) or bool(read_ram_u8(ram, 0x11F02) & 0x80)


def eve_talk_spec(ram):
    """Bar loop once the door is open (17:00, not Sunday). None during the day."""
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    if _eve_married(ram):
        return None
    probe = WorldProbe(ram=ram)
    if probe.is_storm() or probe.weekday() == 0 or probe.clock().hour < 17:
        return None
    hearts = read_ram_value(ram, "eve_hearts")
    if hearts >= _EVE_FLOOR:
        return None
    season, day = probe.calendar_date()
    if season == 0 and day < 23 and hearts >= _EVE_HOLD:
        return None
    if _eve_outlook(ram, probe) in {"dead_zone", "none"}:
        return None
    target = _EVE_HOLD if hearts < _EVE_HOLD else _EVE_FLOOR
    return PhaseSpec(
        "EVE_TALK_LOOP",
        PhaseKind.EVE_TALK_LOOP,
        {"target_points": target, "timeout": 360000},
        failure_policy="optional",
    )


def eve_festival_spec(ram):
    """Spring D23 dance, while the west-edge trigger is still before 15:00."""
    from harvest.planner.day_phase_calendar import attend_festival_phase

    from harvest.planner.power_berry_calendar import snow_seed_due
    seed_due = snow_seed_due(ram)
    if _eve_married(ram) and not seed_due:
        return None
    probe = WorldProbe(ram=ram)
    season, day = probe.calendar_date()
    if (season, day) != (0, 23) or probe.clock().hour >= 15 or probe.is_storm():
        return None
    if read_ram_value(ram, "eve_hearts") >= _EVE_FLOOR and not seed_due:
        return None
    spec = attend_festival_phase(season, day, probe.year())
    return replace(spec, params={**spec.params, "buy_snow_seed": True}) if seed_due else spec


def eve_feather_buy_spec(ram):
    """Sunday peddler. The shelf bit is the receipt; the shed fetch is separate.

    Group 3 event 0 ($B3BA59) always places the intro peddler. It places the
    seller (text 0x01C3 at (568, 360)) only when the first house bit is set,
    a girl has 250–999 hearts, and the purchase flag is still clear.
    """
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    if _eve_married(ram):
        return None
    probe = WorldProbe(ram=ram)
    if (probe.weekday() != 0 or probe.clock().hour >= 16 or probe.is_storm()
            or probe.is_festival()):
        return None
    if read_ram_value(ram, "eve_hearts") < _EVE_FLOOR or read_ram_value(ram, "money") < 1000:
        return None
    if not read_ram_value(ram, "upgrade_flags") & 0x40:
        return None
    if read_ram_value(ram, "family_event_flags") & 0x40:
        return None
    if _feather_owned(ram):
        return None
    return PhaseSpec("BUY_FEATHER", PhaseKind.FEATHER_BUY, failure_policy="optional")


def eve_fetch_spec(ram):
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    if _eve_married(ram) or read_ram_value(ram, "eve_hearts") < _EVE_FLOOR:
        return None
    if tool_in_carry_pair(ram, _FEATHER) or not read_ram_u8(ram, 0x11F02) & 0x80:
        return None
    return PhaseSpec(
        "FETCH_FEATHER", PhaseKind.ENSURE_TOOL, {"tool_id": _FEATHER}, failure_policy="optional",
    )


def eve_propose_spec(ram):
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec
    from harvest.tasks.eve_proposal import proposal_blockers

    if proposal_blockers(ram) is not None or not _feather_owned(ram):
        return None
    return PhaseSpec(
        "PROPOSE_EVE",
        PhaseKind.PROPOSE_EVE,
        {"ends_day": True, "timeout": 60000},
        failure_policy="optional",
    )


def eve_phases_for_ram(ram):
    """Eve work the ordinary day list should add. Empty on a farm morning."""
    from harvest.planner.power_berry_calendar import snow_seed_due
    if snow_seed_due(ram):
        return [eve_festival_spec(ram)]
    propose = eve_propose_spec(ram)
    if propose is not None:
        return [propose]
    return [spec for spec in (
        eve_feather_buy_spec(ram),
        eve_fetch_spec(ram),
        eve_festival_spec(ram),
        eve_talk_spec(ram),
    ) if spec is not None]


def _insert_at(phases, index, extra):
    return list(phases[:index]) + list(extra) + list(phases[index:])


def _festival_index(phases) -> int:
    names = [phase.phase for phase in phases]
    chores = [names.index(name) for name in ("EXIT_COOP", "EXIT_BARN") if name in names]
    if chores:
        return max(chores) + 1
    if "DYNAMIC_OUTDOOR_PLAN" in names:
        return names.index("DYNAMIC_OUTDOOR_PLAN")
    if names[:1] == ["EXIT_TO_FARM"]:
        return 1
    return 0


# The Flower Festival rolls the day (the dance puts the farmer to bed) and the
# west-edge trigger needs town before 15:00, about 700 frames from the farm.
# Watering that stops arming tiles at 12:00 leaves that walk an hour of slack.
FESTIVAL_WATER_STOP_HOUR = 12


def _festival_morning_water(festival, phases):
    """Can and capped watering ahead of a day-ending festival on a farm morning."""
    from harvest.planner.day_phase_catalog import CROP_WATER_PHASE, ENSURE_WATERING_CAN_PHASE

    if not festival.params.get("ends_day") or any(p.phase == "CROP_WATER" for p in phases):
        return []
    if "DYNAMIC_OUTDOOR_PLAN" not in [p.phase for p in phases]:
        return []
    # Optional: a watering failure must never cost the festival (Eve's +42).
    water = replace(CROP_WATER_PHASE, params={**CROP_WATER_PHASE.params, "stop_hour": FESTIVAL_WATER_STOP_HOUR},
                    failure_policy="optional")
    return [ENSURE_WATERING_CAN_PHASE, water]


def _sleep_index(phases) -> int:
    for index, phase in enumerate(phases):
        if phase.phase in {"RETURN_HOME", "GO_TO_SLEEP"}:
            return index
    return len(phases)


_CHORE_PHASES = frozenset({
    "ENSURE_ANIMAL_TOOLS", "ENSURE_BRUSH", "EXIT_TO_FARM", "NAV_TO_BARN", "ENTER_BARN",
    "COW_CHORES", "EXIT_BARN", "NAV_TO_COOP", "ENTER_COOP", "COOP_CHORES", "EXIT_COOP",
})


def _proposal_day_chores(phases):
    """The day list's feed-and-milk phases, to run before the proposal.

    The wedding rolls the day, so a list that jumped straight to it left every
    adult unfed overnight: -8 happiness and a sickness roll each, and a sick cow
    dies seven nights later (Fall D9 of the Fall-D1 replay: cow 0 sick, dead on
    D17, and its removal scene ate D16, so the other cow went unfed as well).
    The proposal window runs to 17:00, so the chores fit before the walk.
    """
    names = [phase.phase for phase in phases]
    ends = [i for i, name in enumerate(names) if name in ("EXIT_COOP", "EXIT_BARN")]
    if not ends:
        return []
    return [p for p in phases[: max(ends) + 1] if p.phase in _CHORE_PHASES]


def apply_eve_phases(phases, ram):
    """Place Eve phases without putting them ahead of the exit on a work morning."""
    from harvest.planner.tasks.home import apply_house_phases
    phases = apply_house_phases(phases, ram)
    social = eve_phases_for_ram(ram)
    if not social:
        return phases
    by_name = {phase.phase: phase for phase in social}
    if "PROPOSE_EVE" in by_name:
        from harvest.core.tile_catalog import ADDR_TILEMAP
        from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE

        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        # The wedding rolls the day, so the list ends on the proposal.
        chores = _proposal_day_chores(phases)
        prefix = chores or ([] if tilemap in _PROPOSE_MAPS else [EXIT_TO_FARM_PHASE])
        fetch = eve_fetch_spec(ram)
        if fetch is not None:
            prefix.append(fetch)
        return prefix + [by_name["PROPOSE_EVE"]]
    out = list(phases)
    if "ATTEND_FESTIVAL" in by_name and not any(phase.phase == "ATTEND_FESTIVAL" for phase in out):
        out = _insert_at(out, _festival_index(out), [*_festival_morning_water(by_name["ATTEND_FESTIVAL"], out),
                                                     by_name["ATTEND_FESTIVAL"]])
    dated_berry = any(p.kind == "harvest_happiness" or (p.kind == "power_berry" and p.params.get("source") in {"harvest_festival", "egg_festival"})
                      for p in out)
    errands = [] if dated_berry else [by_name[name] for name in ("BUY_FEATHER", "FETCH_FEATHER") if name in by_name]
    if errands:
        names = [phase.phase for phase in out]
        at = names.index("EXIT_TO_FARM") + 1 if "EXIT_TO_FARM" in names else 0
        out = _insert_at(out, at, errands)
    if "EVE_TALK_LOOP" in by_name and not any(phase.phase == "EVE_TALK_LOOP" for phase in out):
        from harvest.planner.day_phase_catalog import GO_TO_SLEEP_PHASE, RETURN_HOME_PHASE

        if [phase.phase for phase in out] == ["GO_TO_SLEEP"]:
            return [by_name["EVE_TALK_LOOP"], RETURN_HOME_PHASE, GO_TO_SLEEP_PHASE]
        out = _insert_at(out, _sleep_index(out), [by_name["EVE_TALK_LOOP"]])
    return out


def _nav(hops, *, preserve_held=False, timeout=6000):
    return MultiMapNavTask(waypoints=list(hops), timeout=timeout, initial_settle_frames=30,
                          allow_opportunistic_clear=not preserve_held)


def _south_town_exit():
    return segment_waypoints("town_south_to_path")


def _ann_route(weekday):
    hops = list(segment_waypoints("town_to_shop_door")[:2]) + list(ROUTES["d1_town_to_ann"])
    if weekday == 6:
        return hops + [Waypoint(0x04, (389, 901), radius=0)]
    return hops + [Waypoint(0x04, (408, 914), radius=10),
                   Waypoint(0x04, (408, 888), radius=6, is_exit=True, exit_direction="up")]


@dataclass
class AnnCourtTask(Task):
    """Keep and give two native forage items, checking each heart increment."""

    name: str = "court_ann"
    timeout: int = 16000
    _child: Task | None = field(default=None, init=False)
    _phase: str = field(default="exit", init=False)
    _steps: int = field(default=0, init=False)
    _press: int = field(default=0, init=False)
    _gifts: int = field(default=0, init=False)
    _before: int = field(default=0, init=False)
    _weekday: int = field(default=0, init=False)
    _last: tuple | None = field(default=None, init=False)
    _stasis: int = field(default=0, init=False)
    _gift_item: int = field(default=0, init=False)
    _weed_queue: deque = field(default_factory=deque, init=False)
    _weed_tile: tuple | None = field(default=None, init=False)
    _weed_attempts: int = field(default=0, init=False)
    _align_map: int | None = field(default=None, init=False)

    def reset(self, world):
        self._child = None
        self._phase = "exit"
        self._steps = self._press = self._gifts = self._stasis = 0
        self._last = None
        self._weed_queue.clear()
        self._weed_tile, self._weed_attempts = None, 0
        self._align_map = None
        self._weekday = read_ram_value(world.ram, "weekday")
        self._before = read_ram_value(world.ram, "ann_hearts")

    def can_start(self, world):
        return True

    def _start(self, world, phase, child):
        self._phase, self._child = phase, child
        self._last, self._stasis = None, 0
        child.reset(world)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _pick(self, world):
        if self._gifts:
            from harvest.tasks.mountain_berry import BERRY_NAV_SEGMENTS
            return self._start(world, "pick", MountainBerryTask(
                approach_only=False, pick_attempts=3, stand_px=(40, 57),
                expected_items=frozenset({3}),
                segments=(*BERRY_NAV_SEGMENTS[:2], "mountain_entry_to_second_grape")))
        return self._start(world, "pick", MountainBerryTask(
            approach_only=False, pick_attempts=3, expected_items=frozenset({0x03})))

    def _return_leg(self, world):
        path = (segment_waypoints("path_to_farm") if self._gifts >= 2
                else [Waypoint(0x0C, (132, 128), radius=8)])
        return self._start(world, "return", _nav(_south_town_exit() + path))

    def step(self, world):
        self._steps += 1
        ram = world.ram
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"courting timeout during {self._phase}")
        if self._child is None and self._phase == "exit":
            return self._start(world, "exit", ExitToFarmTask())
        if self._child is not None:
            if (self._phase == "to_ann" and isinstance(self._child, MultiMapNavTask)
                    and self._weekday != 6 and read_ram_value(ram, "tilemap") == 4
                    and read_ram_value(ram, "hour") < 9):
                wp = self._child._current_wp()
                if wp is not None and wp.is_exit and wp.exit_direction == "up":
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="wait for tool shop 09:00")
            queued = drain_action_queue(self._weed_queue)
            if queued is not None:
                return queued
            if self._phase == "pick" and self._gifts and read_ram_value(ram, "tilemap") == 0x10:
                nav = getattr(self._child, "_child", None)
                if isinstance(nav, MultiMapNavTask) and nav._phase == "nav":
                    held = read_ram_value(ram, "held_item")
                    action = read_ram_value(ram, "player_action")
                    if held == 12:
                        if action == 0:
                            self._weed_queue.extend(press_button_sequence("a", pre_press_settle_frames=2, hold_frames=1, settle_frames=24))
                            return drain_action_queue(self._weed_queue)
                        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
                    wp = nav._current_wp()
                    if held == 0 and action == 0 and wp is not None:
                        pos = get_pos_from_ram(ram)
                        face = wp.run_direction or ("right" if wp.target_px[0] > pos.x else "left")
                        dx, dy = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}[face]
                        tile = (pos.x // 16 + dx, pos.y // 16 + dy)
                        if get_tile_at(ram, *tile) == 0x35:
                            # Spring upper-plateau weeds lift as item 12;
                            # toss them before walking, never gift them.
                            if tile != self._weed_tile:
                                self._weed_tile, self._weed_attempts = tile, 0
                            self._weed_attempts += 1
                            if self._weed_attempts > 3:
                                return TaskResult(TaskStatus.FAILURE, reason=f"mountain weed pickup failed at {tile}")
                            self._weed_queue.extend(press_a_sequence(face, face_frames=6, pre_press_settle_frames=2,
                                                                     hold_frames=1, settle_frames=20))
                            return drain_action_queue(self._weed_queue)
            result = self._child.step(world)
            if isinstance(self._child, MultiMapNavTask):
                pos = get_pos_from_ram(ram)
                pixel = (read_ram_value(ram, "tilemap"), pos.x, pos.y)
                free = read_ram_value(ram, "input_lock") == 1 and not read_ram_value(ram, "game_state") & 0x8000
                self._stasis = self._stasis + 1 if free and pixel == self._last else 0
                self._last = pixel
                if self._stasis >= 180:
                    return TaskResult(TaskStatus.FAILURE, reason=f"courting navigation stalled: {self._phase}")
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                return TaskResult(result.status, reason=f"courting {self._phase}: {result.reason}")
            if self._phase == "exit":
                return self._pick(world)
            if self._phase == "pick":
                self._gift_item = read_ram_value(ram, "held_item")
                if self._gift_item not in {1, 3, 4, 5}:
                    return TaskResult(TaskStatus.FAILURE, reason="courting forage not kept")
                path = segment_waypoints("path_mountain_gate_to_farm")[:-1]
                pos = get_pos_from_ram(ram)
                mountain = []
                if self._gifts:
                    mountain = [Waypoint(0x10, px, radius=4, run_direction=face, force_run=True)
                                for px, face in (((56,56), "right"), ((56,136), "down"),
                                                 ((104,136), "right"), ((200,136), "right"),
                                                 ((200,200), "down"), ((304,200), "right"),
                                                 ((408,200), "right"), ((408,264), "down"),
                                                 ((408,328), "down"), ((312,328), "left"),
                                                 ((312,360), "down"))]
                mountain += segment_waypoints("first_berry_to_mountain_exit")
                hops = mountain + path
                hops += segment_waypoints("path_to_town") + _ann_route(self._weekday)
                return self._start(world, "to_ann", _nav(hops, preserve_held=True))
            if self._phase == "to_ann":
                self._before = read_ram_value(ram, "ann_hearts")
                self._phase, self._press = "align", 0
            elif self._phase == "return":
                if self._gifts >= 2 or read_ram_value(ram, "ann_hearts") >= 250 or read_ram_value(ram, "hour") >= 13:
                    if is_path_tilemap(read_ram_value(ram, "tilemap")):
                        return self._start(world, "home", ExitToFarmTask())
                    return TaskResult(TaskStatus.SUCCESS, reason=f"Ann gifts={self._gifts} hearts={read_ram_value(ram, 'ann_hearts')}")
                return self._pick(world)
            elif self._phase == "home":
                return TaskResult(TaskStatus.SUCCESS, reason=f"Ann gifts={self._gifts} hearts={read_ram_value(ram, 'ann_hearts')}")
        if self._phase == "align":
            tilemap = read_ram_value(ram, "tilemap")
            if tilemap != self._align_map:
                self._align_map, self._press = tilemap, 0
            self._press += 1
            pos = get_pos_from_ram(ram)
            stale_room_pos = tilemap in {0x22, 0x23} and (pos.x > 256 or pos.y > 512)
            if (self._press <= 120 or read_ram_value(ram, "game_state") & 0x8000
                    or stale_room_pos or tilemap not in {4, 0x22, 0x23}):
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            if read_ram_value(ram, "held_item") != self._gift_item:
                return TaskResult(TaskStatus.FAILURE, reason="forage lost before Ann stand")
            if self._press > 1200:
                return TaskResult(TaskStatus.FAILURE, reason="Ann stand alignment timeout")
            ann = next((obj for obj in game_objects(ram) if not obj.is_player and 0x0224 <= obj.sprite_table_idx <= 0x0234), None)
            if ann is None:
                return TaskResult(TaskStatus.FAILURE, reason="Ann absent from court stand")
            target = (ann.pixel[0] + 13, ann.pixel[1] + 9)
            tilemap = read_ram_value(ram, "tilemap")
            if tilemap == 0x22 and ann.pixel[1] < 288:
                action = make_action(up=True) if pos.y <= 344 and abs(pos.x - 104) <= 1 else room_walk_action(ram, (104,344))
            elif abs(pos.x - target[0]) <= 1 and abs(pos.y - target[1]) <= 1:
                self._phase, self._press = "gift", 0
                action = make_action()
            elif tilemap in {0x22, 0x23}:
                action = room_walk_action(ram, target)
            elif pos.x != target[0]:
                action = make_action(right=pos.x < target[0], left=pos.x > target[0])
            elif pos.y != target[1]:
                action = make_action(down=pos.y < target[1], up=pos.y > target[1])
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))
        if self._phase == "gift":
            hearts = read_ram_value(ram, "ann_hearts")
            if hearts > self._before:
                delta = hearts - self._before
                if delta not in {5, 6}:
                    return TaskResult(TaskStatus.FAILURE, reason=f"unexpected Ann gift delta {delta}")
                self._gifts += 1
                print(f"[COURT] Ann +{delta} hearts={hearts} gifts={self._gifts}", flush=True)
                if read_ram_value(ram, "tilemap") in {0x22, 0x23}:
                    self._phase, self._press = "leave_ann", 0
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
                return self._return_leg(world)
            self._press += 1
            if self._press > 600 or read_ram_value(ram, "hour") >= 17:
                return TaskResult(TaskStatus.FAILURE, reason="Ann gift did not increase hearts")
            # Stand to her right, face left before offering; A alone throws
            # a held gift if the NPC interaction box is not in front.
            action = make_action(left=True) if self._press <= 6 else make_action(a=self._press % 2 == 0)
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))
        if self._phase == "leave_ann":
            tilemap = read_ram_value(ram, "tilemap")
            if tilemap != self._align_map:
                self._align_map, self._press = tilemap, 0
            self._press += 1
            if tilemap not in {0x22, 0x23}:
                return self._return_leg(world)
            if self._press > 1200:
                return TaskResult(TaskStatus.FAILURE, reason="Ann room exit timeout")
            pos = get_pos_from_ram(ram)
            target = (104,168) if tilemap == 0x23 else (136,440)
            if read_ram_value(ram, "input_lock") != 1:
                action = make_action(a=self._press % 2 == 0)
            elif pos.y >= target[1] and abs(pos.x - target[0]) <= 2:
                action = make_action(down=True)
            else:
                action = room_walk_action(ram, target)
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))


# Seller is opcode 1A at $B3BACD: pixel (0x0238, 0x0168). The intro CC at
# $B3BBC3 is (584, 360) and opens text 0x0078 on every shop-hours talk.
# A waypoint past either sprite sends BFS through the seed-shop door.
_PEDDLER_PX = (568, 360)
_INTRO_PX = (584, 360)
_FEATHER_TEXT = 0x01C3


def _peddler_stand():
    """Stop on the shop-door row. The sprite is south of here; walk, don't BFS."""
    return [Waypoint(0x04, (568, 288), radius=8)]


def _confirm_feather_buy(ram, step: int):
    """Buy it is row 0. $019B bit 0x10 means the two choices are on screen.

    ImmobileInputs reads a new A edge (!key_A = $0080), not a held button.
    A 3-frame press is the bar's pattern. One frame often never latches.
    """
    pressed = step % 12 < 3
    live = len(ram) > 0x019B and bool(int(ram[0x019B]) & 0x10)
    if not live:
        return make_action(a=pressed)
    row = int(read_ram_value(ram, "dialog_choice_result", raw=True))
    if row != 0:
        return make_action(up=pressed)
    return make_action(a=pressed)


def _sprite_at(ram, px, tol: int = 8):
    for obj in game_objects(ram):
        if obj.is_player:
            continue
        ox, oy = obj.pixel
        if max(abs(ox - px[0]), abs(oy - px[1])) <= tol:
            return obj
    return None


def _on_peddler_south(pos) -> bool:
    """Row 23 under the seller. Row 22 is his body, and right into it does not move."""
    return 368 <= int(pos.y) <= 392 and abs(int(pos.x) - _PEDDLER_PX[0]) <= 12


def _skirt_peddler(pos):
    """West around the D3 counter, then east on row 23 under the seller.

    Tile dump from (576, 296): row 21 is D3 across x=35..37, row 22 is the
    sprite row, and F0 (the sell table) is tile x=37. The intro stands on
    tile x=36. The seller's script coordinate is tile x=35.
    """
    sx = _PEDDLER_PX[0]
    x, y = int(pos.x), int(pos.y)
    if y < 368:
        if y < 352 and x > sx - 48:
            return make_action(left=True)
        return make_action(down=True)
    if x < sx - 8:
        return make_action(right=True)
    if x > sx + 8:
        return make_action(left=True)
    return make_action()


def _dump_stand(ram, pos) -> None:
    tx, ty = int(pos.x) // 16, int(pos.y) // 16
    rows = []
    for dy in range(-1, 5):
        cells = [f"{int(get_tile_at(ram, tx + dx, ty + dy)):02X}" for dx in range(-3, 4)]
        rows.append(f"y{ty + dy}:" + " ".join(cells))
    objs = []
    for obj in game_objects(ram):
        if obj.is_player:
            continue
        ox, oy = obj.pixel
        if abs(ox - pos.x) < 160 and abs(oy - pos.y) < 160:
            objs.append(f"0x{obj.sprite_table_idx:04X}@{ox},{oy}")
    print("[FEATHER] tiles " + " | ".join(rows))
    print("[FEATHER] objs " + (" ".join(objs) or "none"))


def _peddler_neighbors(ram) -> str:
    bits = []
    for obj in game_objects(ram):
        if obj.is_player:
            continue
        x, y = obj.pixel
        if abs(x - _PEDDLER_PX[0]) < 96 and abs(y - _PEDDLER_PX[1]) < 128:
            bits.append(f"{obj.label}@{x},{y}")
    return " ".join(bits) or "none"


def _peddler_gate():
    """(744, 424) is already inside (728, 424) radius 18, so that hop never runs.

    BFS from the mouth push-faces (46, 27) and walks back out to the path.
    Left along y=424 is the open face. A north target is how that shove starts.
    """
    return Waypoint(0x04, (700, 424), radius=6, run_direction="left", force_run=True)


def _peddler_town_tail():
    door = segment_waypoints("town_to_shop_door")
    return [_peddler_gate()] + door[1:-1] + _peddler_stand()


def _peddler_after_gate():
    """Hops once the scripted left-hold has cleared x=710. Skip the mouth row."""
    door = segment_waypoints("town_to_shop_door")
    return door[2:-1] + _peddler_stand()


def _peddler_hops_from_farm():
    return list(ROUTES["farm_to_town"]) + _peddler_town_tail()


def _peddler_hops_from_path():
    return list(segment_waypoints("path_to_town")) + _peddler_town_tail()


@dataclass
class FeatherBuyTask(Task):
    """Sunday peddler: walk south from (568, 288) and confirm text 0x01C3."""

    name: str = "buy_feather"
    _child: Task | None = field(default=None, init=False)
    _steps: int = field(default=0, init=False)
    _money: int = field(default=0, init=False)
    _offered: bool = field(default=False, init=False)
    _phase: str = field(default="exit", init=False)
    _last: tuple | None = field(default=None, init=False)
    _stasis: int = field(default=0, init=False)
    _shelf: int = field(default=0, init=False)
    _retries: int = field(default=0, init=False)
    _hold: int = field(default=0, init=False)
    _gate: int = field(default=0, init=False)
    _gate_still: int = field(default=0, init=False)
    _gate_x: int = field(default=0, init=False)
    _dumped: bool = field(default=False, init=False)
    _intro: int = field(default=0, init=False)
    _box: tuple | None = field(default=None, init=False)
    _last_idx: int = field(default=-1, init=False)
    _stall: int = field(default=0, init=False)
    purchase_receipt: dict = field(default_factory=dict, init=False)

    def reset(self, world):
        self._steps = self._stasis = self._retries = self._hold = 0
        self._gate = self._gate_still = self._gate_x = 0
        self._dumped = False
        self._intro = 0
        self._box = None
        self._last_idx = -1
        self._stall = 0
        self._money = read_ram_value(world.ram, "money")
        self._shelf = read_ram_u8(world.ram, 0x11F02)
        self.purchase_receipt.clear()
        self._offered = False
        self._phase = "exit"
        self._last = None
        self._child = ExitToFarmTask()
        self._child.reset(world)

    def can_start(self, world):
        return not read_ram_u8(world.ram, 0x11F02) & 0x80 and self._money >= 1000

    def _begin_talk(self, ram, reason: str):
        pos = get_pos_from_ram(ram)
        text = current_dialogue_registers(ram)["text_id"]
        print(f"[FEATHER] {reason} pos=({pos.x},{pos.y}) text=0x{text:04X} near={_peddler_neighbors(ram)}")
        self._child = None
        self._phase = "talk"
        self._stasis = self._hold = 0
        self._last = None

    def _hold_east_gate(self, world, pos):
        """Script the mouth. MultiMapNav's replan is what steps back onto 0x0C.

        A path-left warp leaks ~(10, 128) on tilemap 0x04 before the real
        stand ~(744, 424). BFS from the leak, or from the mouth, leaves town.
        """
        if self._phase != "nav":
            return None
        ram = world.ram
        if read_ram_value(ram, "tilemap") != 0x04:
            return None
        x, y = int(pos.x), int(pos.y)
        if x <= 200:
            if self._gate != 1:
                print(f"[FEATHER] town entry leak pos=({x},{y})")
                self._gate, self._gate_still = 1, 0
            self._gate_still += 1
            if self._gate_still > 120:
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=f"town entry stayed at ({x},{y})",
                )
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
        if x < 710:
            if self._gate:
                # Warp frames report x=280,y=0 before the real stand. Not clear yet.
                if y < 360:
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
                print(f"[FEATHER] gate clear pos=({x},{y})")
                self._gate = self._gate_still = 0
                self._child = _nav(_peddler_after_gate(), timeout=12000)
                self._child.reset(world)
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            return None
        free = (
            read_ram_value(ram, "input_lock") == 1
            and not read_ram_value(ram, "game_state") & 0x8000
        )
        if self._gate != 2:
            print(f"[FEATHER] gate hold pos=({x},{y})")
            self._gate, self._gate_still, self._gate_x = 2, 0, x
        elif free and x >= self._gate_x:
            self._gate_still += 1
        else:
            self._gate_still = 0
            self._gate_x = x
        if self._gate_still > 150:
            return TaskResult(
                TaskStatus.FAILURE,
                reason=f"east gate left blocked at ({x},{y})",
            )
        nudge = 40 <= self._gate_still < 56
        if nudge and self._gate_still == 40:
            print(f"[FEATHER] gate nudge up pos=({x},{y})")
        up = y > 428 or nudge
        down = y < 416 and not up
        return TaskResult(
            TaskStatus.RUNNING,
            action=ActionResult(make_action(left=True, up=up, down=down, b=True)),
        )

    def step(self, world):
        self._steps += 1
        ram = world.ram
        pos = get_pos_from_ram(ram)
        dialog = current_dialogue_registers(ram)
        text = dialog["text_id"]
        if self._steps > 16000:
            return TaskResult(
                TaskStatus.FAILURE,
                reason=f"feather purchase timeout phase={self._phase} text=0x{text:04X} pos=({pos.x},{pos.y})",
            )
        if read_ram_u8(ram, 0x11F02) & 0x80:
            paid = self._money - read_ram_value(ram, "money")
            if self._offered and paid == 1000:
                self.purchase_receipt = {
                    "frames": self._steps, "text_id": _FEATHER_TEXT,
                    "start": {"money": self._money, "shed_row_3": self._shelf},
                    "end": {"money": read_ram_value(ram, "money"),
                            "shed_row_3": read_ram_u8(ram, 0x11F02)},
                    "wallet_delta": -paid,
                }
                return TaskResult(TaskStatus.SUCCESS, reason="feather bought: 1000G and shelf bit set")
            return TaskResult(TaskStatus.FAILURE, reason=f"unverified feather receipt: paid={paid}")
        if text == _FEATHER_TEXT:
            self._offered = True
            if self._phase != "buy":
                self._begin_talk(ram, "offer")
                self._phase = "buy"
        if self._child is not None:
            if self._phase == "nav":
                held = self._hold_east_gate(world, pos)
                if held is not None:
                    return held
            # The seed-shop door is one tile east of this route. Sitting in
            # its Sunday text is how both buys burned the morning.
            locked = read_ram_value(ram, "input_lock") != 1
            on_door = (read_ram_value(ram, "tilemap") == 0x04 and pos.x > 590 and pos.y < 310)
            if self._phase == "nav" and locked and on_door and text != _FEATHER_TEXT:
                self._hold += 1
                if self._hold >= 30:
                    return TaskResult(
                        TaskStatus.FAILURE,
                        reason=f"shop door dialog 0x{text:04X} at ({pos.x},{pos.y})",
                    )
            elif self._phase == "nav":
                self._hold = 0
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                if self._phase == "nav":
                    pixel = (read_ram_value(ram, "tilemap"), pos.x, pos.y)
                    free = read_ram_value(ram, "input_lock") == 1 and not read_ram_value(ram, "game_state") & 0x8000
                    self._stasis = self._stasis + 1 if free and pixel == self._last else 0
                    self._last = pixel
                    if self._stasis >= 180:
                        return TaskResult(
                            TaskStatus.FAILURE,
                            reason=f"peddler navigation stalled at ({pos.x},{pos.y})",
                        )
                return result
            if result.status != TaskStatus.SUCCESS:
                if (self._phase == "nav" and self._retries < 1
                        and is_path_tilemap(read_ram_value(ram, "tilemap"))):
                    self._retries += 1
                    self._child = _nav(_peddler_hops_from_path(), timeout=12000)
                    self._child.reset(world)
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
                return result
            if self._phase == "exit":
                self._child, self._phase = _nav(_peddler_hops_from_farm(), timeout=12000), "nav"
                self._child.reset(world)
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._begin_talk(ram, "arrived")
            self._phase = "approach"
        if read_ram_value(ram, "hour") >= 17:
            return TaskResult(TaskStatus.FAILURE, reason="peddler window closed")
        pixel = (pos.x, pos.y)
        free = read_ram_value(ram, "input_lock") == 1
        self._stasis = self._stasis + 1 if free and pixel == self._last else 0
        self._last = pixel
        if self._phase == "approach":
            if not self._dumped:
                self._dumped = True
                _dump_stand(ram, pos)
            if _sprite_at(ram, _PEDDLER_PX) is None:
                if self._stasis >= 45:
                    return TaskResult(
                        TaskStatus.FAILURE,
                        reason=f"feather seller not spawned near={_peddler_neighbors(ram)}",
                    )
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            if not free and text not in (0, 0x000A, _FEATHER_TEXT):
                self._hold += 1
                if self._hold == 1:
                    rec = text_record_for_id(text)
                    snippet = ""
                    if rec is not None and rec.text:
                        snippet = rec.text.splitlines()[0][:48]
                    print(f"[FEATHER] other text 0x{text:04X} {snippet}")
                if self._hold > 180:
                    return TaskResult(
                        TaskStatus.FAILURE,
                        reason=f"no feather offer at ({pos.x},{pos.y}) text=0x{text:04X}",
                    )
                return TaskResult(
                    TaskStatus.RUNNING,
                    action=ActionResult(make_action(a=self._steps % 2 == 0)),
                )
            self._hold = 0
            dx = _PEDDLER_PX[0] - pos.x
            dy = _PEDDLER_PX[1] - pos.y
            # (571, 368) is against his southwest corner. Right does not move.
            close = abs(dx) <= 24 and abs(dy) <= 16
            if _on_peddler_south(pos) or (self._stasis >= 16 and close):
                self._begin_talk(ram, "stand")
            elif self._stasis >= 90:
                _dump_stand(ram, pos)
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=f"peddler approach stalled at ({pos.x},{pos.y}) text=0x{text:04X}",
                )
            else:
                return TaskResult(
                    TaskStatus.RUNNING,
                    action=ActionResult(_skirt_peddler(pos)),
                )
        if self._phase == "talk" and text == 0x0078 and self._intro < 3 and not free:
            # Text_Peddler is two pages and about 0x9A words. The id stays
            # 0x0078 until the box closes. Mode 0x02 is the script's box slot.
            # A new A edge turns the page (flags bit 0x08) and closes the end
            # (bit 0x04). Printing itself needs no button.
            self._intro = 1
            self._hold += 1
            flags = int(ram[0x019B]) if len(ram) > 0x019B else 0
            index = int(ram[0x0187]) | (int(ram[0x0188]) << 8) if len(ram) > 0x0188 else 0
            if index != self._last_idx:
                self._last_idx = index
                self._stall = 0
            else:
                self._stall += 1
            snap = (flags & 0x1C, index if flags & 0x1C else self._hold // 120)
            if snap != self._box:
                self._box = snap
                print(
                    f"[FEATHER] intro hold={self._hold} flags=0x{flags:02X} "
                    f"idx=0x{index:04X} lock={dialog['input_lock']}"
                )
            # Bit 0x04 waits up to 120 frames, then the box closes on its own.
            if self._stall > (200 if flags & 0x04 else 160) or self._hold > 900:
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=(
                        f"peddler intro stuck at ({pos.x},{pos.y}) "
                        f"flags=0x{flags:02X} idx=0x{index:04X}"
                    ),
                )
            if self._hold <= 8:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(
                TaskStatus.RUNNING,
                action=ActionResult(make_action(a=self._hold % 12 < 3)),
            )
        if self._phase == "talk" and self._intro == 1 and free:
            print(f"[FEATHER] intro closed pos=({pos.x},{pos.y})")
            return TaskResult(
                TaskStatus.FAILURE,
                reason="peddler intro closed, no feather offer",
            )
        if self._phase == "talk" and text != _FEATHER_TEXT:
            dx = _PEDDLER_PX[0] - pos.x
            dy = _PEDDLER_PX[1] - pos.y
            close = abs(dx) <= 28 and abs(dy) <= 20
            if free and not close and not _on_peddler_south(pos):
                self._phase = "approach"
                self._stasis = self._hold = 0
                return TaskResult(
                    TaskStatus.RUNNING,
                    action=ActionResult(_skirt_peddler(pos)),
                )
            self._hold += 1
            if self._hold > 300:
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=f"no feather offer at ({pos.x},{pos.y}) text=0x{text:04X}",
                )
            if not free:
                return TaskResult(
                    TaskStatus.RUNNING,
                    action=ActionResult(make_action(a=self._steps % 2 == 0)),
                )
            # Tap the facing. Holding a direction walks off into the D3 bin.
            tap = self._steps % 10 < 2
            if abs(dx) >= abs(dy):
                action = make_action(right=dx > 0 and tap, left=dx < 0 and tap, a=self._steps % 8 == 0)
            else:
                action = make_action(down=dy > 0 and tap, up=dy < 0 and tap, a=self._steps % 8 == 0)
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))
        self._phase = "buy"
        # 0x01C3 is the offer. 0x01C4 is the bought line; the wallet and the
        # shelf bit change after that box closes. Each box gets its own budget.
        if text != self._last_idx:
            print(f"[FEATHER] buy text 0x{text:04X}")
            self._last_idx = text
            self._hold = 0
            self._stall = 0
        self._hold += 1
        if free:
            self._stall += 1
            if self._stall > 45:
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=(
                        f"feather dialog closed without receipt text=0x{text:04X} "
                        f"pos=({pos.x},{pos.y})"
                    ),
                )
        else:
            self._stall = 0
            if self._hold > 1200:
                return TaskResult(
                    TaskStatus.FAILURE,
                    reason=f"feather dialog stuck text=0x{text:04X} pos=({pos.x},{pos.y})",
                )
        return TaskResult(
            TaskStatus.RUNNING,
            action=ActionResult(_confirm_feather_buy(ram, self._steps)),
        )
