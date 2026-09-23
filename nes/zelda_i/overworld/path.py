"""Shared hop-path controller for Zelda I overworld level approaches.

Level modules (L2/L3/L5/L6/L8) keep geometry and stop predicates locally;
this module owns the common hop-advance / stuck / swing / maze / door core.
Level 1 remains on the phase-machine in ``overworld/nav.py``. Pre-L1 coast
bombs are ``overworld/shop_p7.py``; the later 0x4A cave is ``bomb_shop.py``.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATES,
    chebyshev,
    direction_to_facing,
    heal_wanted,
    in_sword_hitbox,
    overworld_threat_objects,
)
from zelda_i.dungeon.behaviors import ZORA_SHOT_SPEED, is_projectile
from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE
from zelda_i.dungeon.threat import (
    MIN_DODGE_BODY,
    TRIGGER_TTC,
    Impact,
    ReactiveEvader,
    assess,
    dodgeable,
)
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker, TrackedObject
from zelda_i.overworld.arbiter import Arbiter, Rung
from zelda_i.overworld.common import (
    EDGE_EAST_X,
    EDGE_NORTH_Y,
    EDGE_SOUTH_Y,
    EDGE_WEST_X,
    HEART_FAIRY_DROP_STATES,
    HEART_FAIRY_DROP_TYPES,
    RUPEE_DROP_STATES,
    align_and_push,
    box_step,
    keep_y_band,
    on_arrival_edge,
    perpendicular,
    recover_off_edge,
    scoop_floor_drop,
    shot_escape,
    swing_or_turn,
    track_knockback,
    track_stuck,
    unstick_wiggle,
    wake_or_wait_mode,
    walk_or_swing,
)
from zelda_i.overworld.heart_farm import (
    BAND_SWEEP_WAYPOINTS,
    HeartFarmController,
    HeartFarmPhase,
    owns_bombs,
)
from zelda_i.overworld.graph import (
    MAZE_WAYPOINT_TOL,
    SCREEN_5C_MAZE,
    ScreenHop,
    is_5c_maze_hop,
)
from zelda_i.overworld.hunt import SHOT_DWELL_SPEED, ScreenHunter, blade_lands, hop_lane, link_busy
from zelda_i.overworld.locations import restock_for, worth_heart_farm, worth_rupee_farm
from zelda_i.overworld.rupee_farm import RupeeFarmController, RupeeFarmPhase
from zelda_i.rollout import Rollout, RolloutEvader
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import (
    LATTICE_STEP,
    OccupancyGrid,
    OccupancyWalker,
    WALK_DELTA,
    lattice_route,
    lattice_starts,
    lattice_step,
)

DEFAULT_SWING_PERIOD = 10
DEFAULT_SWING_HOLD = 3
DEFAULT_STUCK_THRESHOLD = 50
DEFAULT_MAX_FRAMES = 30000
DEFAULT_SCOOP_RADIUS = 48
# Hearts and fairies only. Half the play area, so a heal that landed across
# the room is still reachable: a fairy is a full refill and, because one
# ``$0670`` chip takes the sword beam away (``zelda_i.beam``), the heal is
# also the weapon coming back. Rupees keep the short reach — walking 96 px
# for one rupee is how a hop table turns into a farm loop.
DEFAULT_SCOOP_HEAL_RADIUS = 96
_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
# Dungeon OccupancyGrid xmax=216 traps OW east-mouth leftover x≈240.
_OW_OCC_BOUNDS = (0, 255, 0, 239)
_ALIGN_X_TOL = 5
# One pixel inside each EDGE_* so every evade step stays on this screen.
_EVADE_BOUNDS = (EDGE_WEST_X + 1, EDGE_EAST_X - 1, EDGE_NORTH_Y + 1, EDGE_SOUTH_Y - 1)
# Ban dirs this far from a scroll line so a 10-frame commit cannot leave.
_EVADE_EDGE_MARGIN = 12
# Evader only peels an in-pad body after this many stuck frames.
_HEXISH = re.compile(r"(?:0x)?[0-9a-f]{2}")
_TRAILING_DIGITS = re.compile(r"(?<=[a-z])\d+")
_WEDGE_STUCK_FRAMES = 24
# Frames one hop may spend on one screen before the optional rungs are
# switched off and the push is all that is left. Crossing a coast screen is
# ~220 frames of walking, so this is an order of magnitude of slack for
# fighting, scooping and dodging — and still an order of magnitude below the
# 30000 frame stage cap that used to be the only thing to hit.
DEFAULT_HOP_SCREEN_MAX_FRAMES = 4000
# How close an inbound shot has to be before it outranks a body already in
# the blade box. ``threat.TRIGGER_TTC`` is the evader's own trigger, so
# anything looser would drop the swing for a shot the evader is not yet
# acting on; anything tighter and the sidestep cannot finish (Link walks
# 1 px/frame and the pad is ``LINK_HALF + SHOT_HALF``).
_SHOT_OVER_SWORD_TTC = TRIGGER_TTC
_OCCUPIED_LANE_STAND_CAP = 8  # then yield the hop
# Perp walk before writing off the parallel lane; then own-row / stand / yield.
_OCCUPIED_LANE_STEER_CAP = 48
# Lane frames this hop may spend on one screen without gaining a pixel toward
# its exit. ``_OCCUPIED_LANE_STEER_CAP`` cannot see the failure it was written
# for: the peel and the push alternate frames, and every push frame takes the
# ``not blocked`` early return, which zeroes the steer counter. Live 0x7C
# (``scratch/probe_walk_trace.py`` w1): 1795 lane frames and 1982 hop frames
# held Link between x=24 and x=25 at y=109 for 3593 frames — the peel steps
# LEFT, the push steps RIGHT, and neither counter ever reaches its cap. Only
# travel-axis *progress* separates a peel that is going around a body from a
# tug-of-war, so that is what resets this one. Link walks 1 px/frame, so 120
# is a full vertical traverse of the screen with nothing to show for it.
_OCCUPIED_LANE_NO_GAIN_CAP = 120
# Committed stall escape; ``stuck`` reset must not hand the frame back to hop.
_STALL_ESCAPE_COMMIT_FRAMES = 600
# Chebyshev gap at which an unblockable shot outranks every other rung.
#
# The Zora is the live case and it is not a fight: ``prey.SKIP_TYPES`` never
# chases one, ``beam`` never shoots one, and ``behaviors.shield_blocks`` says
# the small shield does not stop its ``0x55`` spit. So the whole policy is
# *dodge and run*, and ``scratch/zora1.json`` says when the dodge has to
# start: the shot is aimed at **launch** and then flies straight at ~1.5
# px/frame, so no step taken before the mouth opens survives it and no swing
# taken after it shortens the contact. Link walks 1 px/frame and the pad is
# ``MIN_DODGE_SHOT`` (12), so a 96 px closing gap is ~64 frames of flight —
# five times the walk the step needs, and short enough that a shot already
# crossing the far side of the screen does not own the hop.
_SPIT_DUCK_RADIUS = 96
# Hold one escape direction while the same shot is still closing. The bearing
# rotates as a shot goes past, so re-deciding every frame is the two-pixel
# tug-of-war this module keeps re-learning (see ``_OCCUPIED_LANE_NO_GAIN_CAP``).
_SPIT_DUCK_COMMIT = 12
# Frames a duck may press a direction that moves Link nowhere before that
# direction is written off as wall for the rest of the screen.
_SPIT_DUCK_STILL_CAP = 4
# A body this close outranks the shot: the duck is a walk, and a walk inside
# a body's own dodge pad cannot clear it — ``MIN_DODGE_BODY`` is the distance
# a sidestep needs before it has moved Link out of a hitbox at all, so inside
# it the honest answers are the blade and the peel. A body further out than
# the shot still yields to the shot: that is ``evade_shot_over_sword``, and a
# 0x55 is the one thing no swing shortens.
_SPIT_DUCK_BODY_PAD = MIN_DODGE_BODY
# Wall memory resolution. A rock blocks a place, not a screen.
_DUCK_CELL = 16
# Frames one body may hold the parry before the evader's step gets them back.
_PARRY_SLOT_MAX_FRAMES = 90

# --------------------------------------------------- the hop ladder ---
# ``_do_hop`` used to be a chain of ``if act is not None: return act`` and its
# precedence was therefore source-line order — unnameable, unassertable, and
# only movable by moving a line. These are the same rungs in the same order,
# as numbers (``overworld.arbiter``: priority counts *down*, 0 is the top and
# the lowest number that claims the frame wins).
#
# They are spaced by ten so a rung can be slotted between two neighbours
# without renumbering the ladder, and so a subclass can express "just below
# that one" as ``+ 1`` (see ``ShopP7WalkController.extra_hop_priority``).
#
# ``align_and_push`` is deliberately *not* a rung: it is the fall-through the
# whole ladder declining resolves to, which is exactly ``Arbiter.decide``
# returning ``None``.
HOP_RUNG_EXTRA = 10  # ``_extra_hop_action``, the subclass hook
HOP_RUNG_MAZE = 20
# The shot. The ladder position is the whole point of this rung: AGENTS.md
# Traps, "travelling frames must offer ``ScreenHunter.take_beam`` above
# stall-escape (600f commits used to zero the weapon)". It used to be an
# out-of-band call into ``self.hunter.take_beam`` from ``_do_hop``, made
# there purely because a hop had no way to say "this rung goes above that
# one". It is now a number, and ``HOP_RUNG_BEAM < HOP_RUNG_STALL_ESCAPE`` is
# a test (``test_arbiter``), not a comment.
HOP_RUNG_BEAM = 30
HOP_RUNG_SCOOP = 40
HOP_RUNG_OCCUPANCY = 50
HOP_RUNG_STALL_ESCAPE = 60
HOP_RUNG_UNSTICK = 70
HOP_RUNG_EDGE = 80
HOP_RUNG_HUNT = 90
# The measured route (``_rung_geo``): the blind align below it walks the
# current row into rock whenever a knockback leaves Link off the planned row
# (0x1E ``letter``: 2528 frames of ``unstick_wait`` against the x=96..143
# block). Below the hunt, so a chase still owns its screen; above the lane
# and the fall-through align, which are the two it replaces when blocked.
HOP_RUNG_GEO = 95
HOP_RUNG_LANE = 100

# ------------------------------------------------ the threat ladder ---
# ``_threat_action`` runs *above* the whole hop ladder: it is consulted in
# ``_decide`` before ``_do_hop`` is reached at all, and it was the same
# ``if act is not None: return act`` chain the hop used to be. These are its
# rungs as numbers, on the same ``Arbiter`` and with the same convention
# (priority counts down; the lowest number that claims the frame wins).
#
# The duck stays on top. It is the one rung that answers a ``0x55``, it is
# live on the M5 Clean chain at 18909f, and ``AGENTS.md`` Traps spells its
# place out ("above the evader and the hunt"). It also declines exactly the
# window the rollout below it is written for: ``TrackedObject.closing_on``
# is False for a shot that has not moved, so a spit still sitting on the
# Zora's muzzle is invisible to the duck *and* to ``assess``.
THREAT_RUNG_DUCK = 10
# The ROM-truth evader (``rollout.RolloutEvader``). Off unless
# ``attach_rollout`` has bound it, so the default ladder is byte-for-byte the
# chain that was here before: duck, then reactive. It is deliberately a
# *sibling* of the reactive rung rather than a replacement for it -- both
# stay in the tree and both are selectable from the same pin, which is the
# whole point (a rollout cannot be scored against a model that is not there).
THREAT_RUNG_ROLLOUT = 20
# ``threat.ReactiveEvader`` plus the yield/in-pad gates it runs behind. The
# default arm, and the fall-back for every frame the rollout declines. The
# rollout does not inherit the sword-yield gate: standing still is one of its
# candidates, so a safe stand declines to this rung naturally, while an
# otherwise-invisible muzzle hit is still allowed to beat the swing.
THREAT_RUNG_REACTIVE = 30


def _step_into_body(
    lx: int, ly: int, direction: str, bodies: tuple[Any, ...]
) -> bool:
    """True when walking ``direction`` a full pad lands inside a body."""
    dx, dy = _DUCK_STEP.get(direction, (0, 0))
    nx, ny = lx + dx * MIN_DODGE_BODY, ly + dy * MIN_DODGE_BODY
    return any(
        chebyshev(nx, ny, int(b.x), int(b.y)) < MIN_DODGE_BODY for b in bodies
    )


_DUCK_STEP = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def _reason_key(reason: str) -> str:
    """Stem of a ``FrameAction.reason``, so a census has ~20 rows not ~2000.

    Reasons carry the hop index and the screen id (``hop4``, ``hunt_7b``,
    ``79_skirt_beach``) — exactly the detail a *per-screen* census already
    supplies, and enough to shatter every row into a singleton if it is kept.
    Both are dropped: trailing digits on a token, and a whole token that is
    just two hex digits.
    """
    stem = str(reason or "none").split("|", 1)[0]
    stem = _TRAILING_DIGITS.sub("", stem)
    kept = [p for p in stem.split("_") if p and not _HEXISH.fullmatch(p)]
    return "_".join(kept) or stem or "none"


def _ow_hop_grid() -> OccupancyGrid:
    xmin, xmax, ymin, ymax = _OW_OCC_BOUNDS
    return OccupancyGrid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _pad_hits(x: int, y: int, hazards: tuple[TrackedObject, ...]) -> bool:
    pad = MIN_DODGE_BODY
    return any(abs(int(h.x) - x) < pad and abs(int(h.y) - y) < pad for h in hazards)


def _lane_blocked(x: int, y: int, direction: str, hazards: tuple[TrackedObject, ...]) -> bool:
    dx, dy = WALK_DELTA.get(direction, (0, 0))
    return any(_pad_hits(x + dx * t, y + dy * t, hazards) for t in range(1, MIN_DODGE_BODY + 1))


def _parallel_lane(
    x: int, y: int, direction: str, hazards: tuple[TrackedObject, ...],
    banned: set[str], *, toward: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Nearest clear parallel-lane cell. ``toward`` only breaks the side tie."""
    xmin, xmax, ymin, ymax = _EVADE_BOUNDS
    near = min(hazards, key=lambda h: abs(int(h.x) - x) + abs(int(h.y) - y))
    if direction in ("LEFT", "RIGHT"):
        perp = ("DOWN", "UP") if y > int(near.y) else ("UP", "DOWN")
        if toward is not None and int(toward[1]) != y:
            perp = ("DOWN", "UP") if int(toward[1]) > y else ("UP", "DOWN")
    else:
        perp = ("RIGHT", "LEFT") if x > int(near.x) else ("LEFT", "RIGHT")
        if toward is not None and int(toward[0]) != x:
            perp = ("RIGHT", "LEFT") if int(toward[0]) > x else ("LEFT", "RIGHT")
    for dist in range(1, MIN_DODGE_BODY * 2 + 1):
        for name in perp:
            if name in banned:
                continue
            px, py = WALK_DELTA[name]
            nx, ny = x + px * dist, y + py * dist
            if (xmin <= nx <= xmax and ymin <= ny <= ymax
                    and not _pad_hits(nx, ny, hazards)
                    and not _lane_blocked(nx, ny, direction, hazards)):
                return (nx, ny)
    return None


class PathNavPhase(Enum):
    """Generic hop-path phases. Levels may use their own HOP/DONE/FAILED/DOOR."""

    HOP = auto()
    DOOR = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldPathController:
    """Frame policy: walk a ``ScreenHop`` table, optional maze, optional door.

    Subclasses override ``_at_stop``, ``_after_hops``, ``_before_play``, and
    ``_extra_hop_action``. Phase enums may differ; helpers resolve by name.
    """

    hops: tuple[ScreenHop, ...] = ()
    hop_index: int = 0
    phase: Any = PathNavPhase.HOP
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    # Knockback: mode 8 keeps Link moving, so hits are counted, not frames.
    last_health: int = -1
    hits_taken: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    swing_period: int = DEFAULT_SWING_PERIOD
    swing_hold: int = DEFAULT_SWING_HOLD
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD
    max_frames: int = DEFAULT_MAX_FRAMES
    allowed_modes: frozenset[int] = field(
        default_factory=lambda: frozenset({PLAY_MODE, 8, 11})
    )

    # Maze. Default pred is is_5c_maze_hop when waypoints set.
    maze_hop_pred: Callable[[ScreenHop], bool] | None = None
    maze_waypoints: tuple[tuple[int, int], ...] = ()
    maze_wp_index: int = 0
    maze_screen: int = SCREEN_5C_MAZE
    maze_tol: int = MAZE_WAYPOINT_TOL

    # Door hunt after hops complete
    door_x: int | None = None
    door_dir: str = "UP"
    door_screen: int | None = None
    entry_level: int | None = None
    entry_room: int | None = None
    require_dungeon: bool = False
    require_entrance_screen: bool = False

    # Low-heart recovery. ``farm_below_hearts=0`` disables (old run-to-death).
    farm_below_hearts: int = 3
    farm_min_filled: int = 3
    farm_max_frames: int = 3600
    max_farm_attempts: int = 2
    farm_attempts: int = 0
    _farm: HeartFarmController | None = field(default=None, repr=False)

    # Restock-farm target. 0 disables the farm loop. Floor scoop is
    # ``scoop_rupees`` or (need_rupees > 0 and still short).
    need_rupees: int = 0
    scoop_rupees: bool = False
    scoop_bombs: bool = False
    hop_screen_max_frames: int = DEFAULT_HOP_SCREEN_MAX_FRAMES
    scoop_radius: int = DEFAULT_SCOOP_RADIUS
    scoop_heal_radius: int = DEFAULT_SCOOP_HEAL_RADIUS
    # In-route kill+restock so we arrive at the shop closer to ``need_rupees``.
    rupee_farm_attempts: int = 0
    _rupee_farm: RupeeFarmController | None = field(default=None, repr=False)
    # Leftover-relative column walk for UP/DOWN hops with align_x.
    _hop_walker: OccupancyWalker | None = field(default=None, repr=False)
    _hop_walker_key: tuple[int, int] | None = field(default=None, repr=False)

    # Measured overworld route (``_rung_geo``). Needs ``bind_env``; with no
    # env bound the rung declines and the hop is the blind align it was.
    geo: bool = True
    # The spit duck scored on the lattice (``_spit_escape``); needs ``geo``.
    shot_model: bool = True
    # Rocks too. The small shield blocks an octorok rock only while Link faces
    # it and is not swinging, so a rock is a shot to walk off like the spit.
    shot_model_rocks: bool = True
    _env: Any = field(default=None, repr=False)
    _geo_nodes: frozenset[tuple[int, int]] | None = field(default=None, repr=False)
    _geo_key: tuple[int, int] | None = field(default=None, repr=False)
    _geo_route: list[tuple[int, int]] | None = field(default=None, repr=False)
    _geo_route_key: tuple[int, int, int] | None = field(default=None, repr=False)
    geo_frames: int = 0

    # Off by default. On, it runs *ahead* of the hop rules.
    evade: bool = False
    _parry_key: tuple[int, int, int] | None = field(default=None, repr=False)
    _parry_frames: int = 0
    evades: int = 0
    parries: int = 0
    evade_reasons: dict[str, int] = field(default_factory=dict)
    # Frames per behaviour per screen, keyed by the stem of the action's
    # ``reason``. Accounting only; nothing reads it to decide anything.
    reason_by_screen: dict[str, dict[str, int]] = field(default_factory=dict)
    # Shorter velocity window for shots only (``ObjectTracker.shot_history``).
    # ``None`` keeps the six-sample read every other hop table is measured
    # against; the hunting walk sets 2 because the Zora's muzzle hold smears
    # its own spit into a standing object for the length of the dodge window.
    shot_history: int | None = None
    _tracker: ObjectTracker | None = field(default=None, repr=False)
    _evader: ReactiveEvader | None = field(default=None, repr=False)
    # The other arm. ``rollout.RolloutEvader`` answers the same question off
    # the ROM instead of off a straight line, and it is opt-in: nothing
    # constructs it, because it needs the *live* env (one emulator per
    # process) and a ``ZeldaSnapshot`` is not one. ``attach_rollout`` is the
    # only constructor, so a caller selects the arm and nothing else moves.
    rollout_evade: bool = False
    _rollout: RolloutEvader | None = field(default=None, repr=False)
    # Built once from ``threat_rungs``; the rungs read ``self.hunter`` at call
    # time, so unlike the hop ladder this one never needs a rebuild.
    _threat_arbiter: Arbiter | None = field(default=None, repr=False)
    # The frame's hop, for the rungs. Same trick as ``_hop``: a rung is
    # ``(snap) -> FrameAction | None``, so the second argument
    # ``_threat_action`` used to thread down its chain is set once a frame.
    _threat_hop: Any = field(default=None, repr=False)
    _tracked: tuple[TrackedObject, ...] = field(default=(), repr=False)
    _evade_room: tuple[int, int] | None = field(default=None, repr=False)
    # The committed spit dodge: direction, the slot it is dodging, and how
    # many frames it has held. ``spit_ducks`` is accounting only.
    spit_ducks: int = 0
    _duck_dir: str | None = field(default=None, repr=False)
    _duck_slot: int | None = field(default=None, repr=False)
    _duck_frames: int = field(default=0, repr=False)
    # ``(direction, cell)`` pairs this screen has *measured* to be wall, and
    # the frames the current one has failed to move Link. ``_EVADE_BOUNDS``
    # is a rectangle; the coast is not. The key carries the cell because a
    # rock blocks a *place* — writing the direction off screen-wide poisons
    # the dodge for the rest of the wave (``zfixB``: 0x7B wrote off LEFT and
    # RIGHT, then took eight hits on one screen).
    _duck_walls: set[tuple[str, int, int]] = field(default_factory=set, repr=False)
    _duck_xy: tuple[int, int] | None = field(default=None, repr=False)
    _duck_still: int = field(default=0, repr=False)
    occupied_lane: bool = False  # L2 on; travel cell in a body pad → parallel lane
    _lane_stand: int = 0
    _lane_steer: int = 0
    # Per ``(hop_index, screen)``: the best travel-axis pixel the lane branch
    # has seen, the frames it has owned since that pixel, and the latch that
    # writes the branch off for the rest of this hop's visit.
    _lane_key: tuple[int, int] | None = field(default=None, repr=False)
    _lane_gain: int | None = field(default=None, repr=False)
    _lane_nogain: int = field(default=0, repr=False)
    _lane_off: bool = field(default=False, repr=False)
    _hop_screen: tuple[int, int] | None = field(default=None, repr=False)
    _hop_screen_frames: int = field(default=0, repr=False)

    # Where ``_extra_hop_action`` sits on the hop ladder. It is a *number*
    # rather than the position of the call site, which is the whole of the
    # 0x79 inversion (``overworld.arbiter`` docstring): a hook pinned to the
    # top of the chain can only decline its way downwards, and a decline
    # written to let the rung below have the frame is a precedence edit
    # spelled as a read of another module's bookkeeping.
    extra_hop_priority: int = HOP_RUNG_EXTRA
    # Built once from ``hop_rungs``; rebuilt if the hunter is swapped, since
    # the beam rung is the hunter's own bound ``take_beam``.
    _hop_arbiter: Arbiter | None = field(default=None, repr=False)
    _hop_arbiter_hunter: Any = field(default=None, repr=False)
    # The frame's hop and its grind verdict, read by the rungs. The rung
    # signature is ``(snap) -> FrameAction | None`` on purpose (it is the
    # shape every ``if act is not None`` branch already had), so the two
    # extra arguments ``_do_hop`` used to pass down its chain are set once a
    # frame here rather than threaded through ten closures.
    _hop: ScreenHop | None = field(default=None, repr=False)
    _hop_grinding: bool = field(default=False, repr=False)

    # Off by default. Hunt runs *after* evade and *after* edge recovery.
    # The hunter is a collaborator, not a flag bundle: whoever builds the path
    # decides which screens it crosses and whether a re-entry reopens them
    # (``ScreenHunter.transit_screens`` / ``reopen_on_enter``), because those
    # are answers about the fight, not about the hop table. ``None`` is a path
    # that never fights, which is every hop table but ``shop_p7``'s.
    hunter: ScreenHunter | None = None
    # Also hunt the screen the hop table ends on (scroll-in wave). This one is
    # path policy, not hunter config — it decides whether the hop table running
    # out ends the walk or starts one more fight — so it stays here with
    # ``_wants_post_hop`` / ``_final_hunt`` and off ``ScreenHunter``.
    hunt_destination: bool = False
    escape_commit_frames: int = _STALL_ESCAPE_COMMIT_FRAMES
    stall_escapes: int = 0
    _escape_frames: int = field(default=0, repr=False)
    _escape_screen: int = field(default=-1, repr=False)

    require_sword: bool = False
    require_triforce_bit: int | None = None
    stop_y_lo: int = 40
    stop_y_hi: int = 210

    def _phase_member(self, name: str) -> Any:
        return type(self.phase)[name]

    def _set_phase(self, phase: Any, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.stuck = 0
            if note:
                self.notes.append(note)

    def _set_phase_name(self, name: str, note: str = "") -> None:
        self._set_phase(self._phase_member(name), note)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        return walk_or_swing(
            self.phase_frames,
            direction,
            reason,
            getattr(self, "_nav_snap", None),
            period=self.swing_period,
            hold=self.swing_hold,
        )

    def _finish(self, note: str = "path_stop") -> FrameAction:
        self.success = True
        self._set_phase_name("DONE", note)
        return FrameAction(nes_idle_action(), "done")

    def _fail(self, note: str) -> FrameAction:
        self._set_phase_name("FAILED", note)
        return FrameAction(nes_idle_action(), note)

    def reset(self) -> None:
        self.hop_index = 0
        self.phase = self._phase_member("HOP")
        self.frames = self.phase_frames = self.stuck = 0
        self.last_x = self.last_y = self.last_screen = self.last_health = -1
        self.hits_taken = 0
        self.farm_attempts = self.rupee_farm_attempts = 0
        self._farm = self._rupee_farm = None
        self._hop_walker = self._hop_walker_key = None
        self._geo_nodes = self._geo_key = None
        self._geo_route = self._geo_route_key = None
        self.geo_frames = 0
        self.evades = self.parries = 0
        self.evade_reasons = {}
        self.reason_by_screen = {}
        self._tracker = self._evader = None
        self._tracked = ()
        # The kernel was handed in (it holds the live env), so it is not ours
        # to drop -- the same rule the hunter gets two blocks down.
        if self._rollout is not None:
            self._rollout.reset()
        self._threat_arbiter = None
        self._threat_hop = None
        self._evade_room = None
        self.spit_ducks = 0
        self._duck_dir = self._duck_slot = None
        self._duck_frames = 0
        self._parry_key, self._parry_frames = None, 0
        self._duck_walls = set()
        self._duck_xy = None
        self._duck_still = 0
        self._lane_stand = self._lane_steer = 0
        self._lane_key = self._lane_gain = None
        self._lane_nogain = 0
        self._lane_off = False
        self._hop_screen = None
        self._hop_screen_frames = 0
        # Dropping the ladder rather than ``Arbiter.reset``-ing it also picks
        # up a hunter swapped in since the last walk (the beam rung is that
        # hunter's bound method).
        self._hop_arbiter = None
        self._hop_arbiter_hunter = None
        self._hop = None
        self._hop_grinding = False
        if self.hunter is not None:
            # The hunter was handed in, so it is not ours to drop: dropping it
            # would silence the fight, where rebuilding it from the old flags
            # only cleared its census. ``ScreenHunter.reset`` leaves the
            # transit/reopen policy alone, which is the same thing.
            self.hunter.reset()
        self.stall_escapes = 0
        self._escape_frames = 0
        self._escape_screen = -1
        self.success = False
        self.notes.clear()
        self.maze_wp_index = 0

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            current = self.hops[self.hop_index]
            hop = {
                "index": self.hop_index,
                "target": current.target,
                "direction": current.direction,
            }
            if self._is_maze_hop(current):
                hop["maze"] = True
        out: dict[str, Any] = {
            "success": self.success,
            "phase": self.phase.name if hasattr(self.phase, "name") else str(self.phase),
            "frames": self.frames,
            "hop_index": self.hop_index,
            "hop": hop,
            "notes": list(self.notes),
            "stuck": self.stuck,
            "hits_taken": self.hits_taken,
            "farm_attempts": self.farm_attempts,
            "rupee_farm_attempts": self.rupee_farm_attempts,
            "need_rupees": self.need_rupees,
        }
        out["reason_by_screen"] = {
            screen: dict(sorted(rows.items(), key=lambda kv: -kv[1]))
            for screen, rows in self.reason_by_screen.items()
        }
        # ``reason_by_screen`` answers "which reason string", which is not the
        # same question as "which behaviour owned these frames": one rung
        # emits several reasons (``hop0``, ``hop0_lane``, ``79_skirt_beach``)
        # and one reason can come from two rungs. The census is per *rung*,
        # so a run report can price a behaviour, and ``push`` is the frames
        # the whole ladder declined — the plain ``align_and_push``.
        ladder = self.hop_arbiter
        out["rung_census"] = dict(ladder.census(), push=ladder.idle_frames)
        out["rung_frames"] = ladder.frames
        if self.evade:
            # The other ladder. It sits *above* the hop one (``_decide`` asks
            # it before ``_do_hop`` exists), so its frames are not in
            # ``rung_census`` at all and an A/B between the two evaders is
            # unreadable without it. ``yielded`` is the frames every threat
            # rung declined — the quiet frames the hop then drove.
            threat = self.threat_arbiter
            out["threat_census"] = dict(
                threat.census(), yielded=threat.idle_frames
            )
            out["threat_frames"] = threat.frames
        if self._rollout is not None:
            # A rollout is a ``set_state``. It is not the kind
            # ``docs/STATUS.md`` means by an assist — the walk's own tape is
            # bit-identical either side of a fan — but a report that hides it
            # is lying about the walk, so the ledger rides next to the census
            # that prices it.
            out["rollout"] = self._rollout.report()
        if self.hunter is not None:
            out["hunt"] = self.hunter.report()
            out["kills"] = self.hunter.kills
            out["stall_escapes"] = self.stall_escapes
        if self.evade:
            out["evades"] = self.evades
            out["parries"] = self.parries
            out["spit_ducks"] = self.spit_ducks
            out["evade_reasons"] = dict(self.evade_reasons)
        if self.maze_waypoints:
            out["maze_wp_index"] = self.maze_wp_index
        if self.require_dungeon or self.require_entrance_screen:
            out["require_dungeon"] = self.require_dungeon
            out["require_entrance_screen"] = self.require_entrance_screen
        return out

    def _wants_post_hop(self) -> bool:
        """True when hops complete should continue (door hunt / dest hunt).

        Without ``hunt_destination``, ``_on_hop_advanced`` finishes on scroll-in.
        """
        return self.require_dungeon or self.require_entrance_screen or (
            self.hunter is not None and self.hunt_destination
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon and self.entry_level is not None:
            if snap.level != self.entry_level or snap.mode != PLAY_MODE:
                return False
            if self.entry_room is not None and snap.screen != self.entry_room:
                return False
            return True
        if self.require_entrance_screen and self.door_screen is not None:
            if not (snap.level == 0 and snap.mode == PLAY_MODE
                    and snap.screen == self.door_screen):
                return False
            if self.require_sword and not snap.has_sword:
                return False
            if self.require_triforce_bit is not None:
                if not (snap.triforce & self.require_triforce_bit):
                    return False
            return True
        end_screen = (
            self.hops[-1].target if self.hops
            else (self.door_screen if self.door_screen is not None else -1)
        )
        if end_screen < 0:
            return False
        if not (self.hop_index >= len(self.hops) and snap.level == 0
                and snap.mode == PLAY_MODE and snap.screen == end_screen
                and self.stop_y_lo < snap.link_y < self.stop_y_hi):
            return False
        if self.require_sword and not snap.has_sword:
            return False
        if self.require_triforce_bit is not None:
            if not (snap.triforce & self.require_triforce_bit):
                return False
        return True

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        """Default post-hop: align door_x and push door_dir; idle in dungeon."""
        if self.entry_level is not None and snap.level == self.entry_level:
            return FrameAction(nes_idle_action(), "dungeon_settle")
        if self.door_x is not None and abs(snap.link_x - self.door_x) > 5:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing(self.door_dir, "door_hunt")

    def destination_hunted(self, snap: ZeldaSnapshot) -> bool:
        """True when the last screen's wave no longer holds the stage open.

        The guard clause is not an optimisation. ``_after_hops`` answers a
        declining hunt with an *idle*, and ``ScreenHunter.step`` declines for
        the whole guard branch once Link is at ``min_hearts`` — so on the
        destination screen those two meet as a stand, for
        ``HUNT_DESTINATION_FRAMES`` (2400). ``pre_l1_anyrow1`` reached 0x6F
        for the first time and died there on frame 593 of that stand, to the
        screen's own Zora, with the cave mouth two tiles away. The
        destination wave is a rupee errand; at one heart it is not worth a
        2400 frame stand, and the stage after this one is the buy.
        """
        if self.hunter is None or not self.hunt_destination:
            return True
        if int(snap.whole_hearts) <= int(self.hunter.min_hearts):
            return True
        return int(snap.screen) in self.hunter.done

    def _final_hunt(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Clear the destination screen before the post-hop policy drives."""
        if self.hunter is None or not self.hunt_destination:
            return None
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return None
        return self.hunter.take_destination(snap, self.frames)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        final = self._final_hunt(snap)
        if final is not None:
            return final
        if self._wants_post_hop():
            if "DOOR" in type(self.phase).__members__:
                if self.phase.name == "HOP":
                    self._set_phase_name("DOOR", "door_hunt")
            return self._simple_door_hunt(snap)
        return self._finish("hops_complete")

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Hook before hop/door play logic (e.g. exit wrong dungeon)."""
        return None

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """Hook after advance check, before maze/align (e.g. 0x5B north corridor)."""
        return None

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.hop_index < len(self.hops):
            return FrameAction(nes_action(self.hops[self.hop_index].direction), "scroll")
        return FrameAction(nes_idle_action(), "scroll_idle")

    def _is_maze_hop(self, hop: ScreenHop) -> bool:
        pred = self.maze_hop_pred
        if pred is None:
            if self.maze_waypoints:
                pred = is_5c_maze_hop
            else:
                return False
        return pred(hop)

    def _in_maze_phase(self, snap: ZeldaSnapshot, hop: ScreenHop) -> bool:
        if not self._is_maze_hop(hop) or not self.maze_waypoints:
            return False
        return snap.screen == self.maze_screen

    def _advance_hop(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """If arrived off the entry edge, advance hop index. Return action if handled."""
        if (snap.screen != hop.target or snap.mode not in (PLAY_MODE, 8)
                or snap.transitioning or on_arrival_edge(hop.direction, snap)):
            return None
        self.notes.append(f"hop_{self.hop_index}_{hop.target:02x}")
        self._escape_frames = 0
        if self._is_maze_hop(hop):
            self.notes.append("maze_complete")
        self.hop_index += 1
        self.stuck = 0
        self.hits_taken = 0
        self.phase_frames = 0
        self.maze_wp_index = 0
        return self._on_hop_advanced(snap, hop)

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        """Called after hop_index increments. Default: done or idle advance."""
        if self.hop_index >= len(self.hops) and not self._wants_post_hop():
            return self._finish("path_complete")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _follow_maze(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self.maze_waypoints:
            return self._swing("RIGHT", "maze_no_waypoints")
        if "maze_start" not in self.notes:
            self.notes.append("maze_start")
        if self.maze_wp_index >= len(self.maze_waypoints):
            return self._swing("RIGHT", "maze_exit")
        tx, ty = self.maze_waypoints[self.maze_wp_index]
        if (abs(snap.link_x - tx) <= self.maze_tol
                and abs(snap.link_y - ty) <= self.maze_tol):
            self.maze_wp_index += 1
            self.stuck = 0
            if self.maze_wp_index >= len(self.maze_waypoints):
                return self._swing("RIGHT", "maze_exit")
            tx, ty = self.maze_waypoints[self.maze_wp_index]
        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="maze_unstick")
            return action
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) > self.maze_tol:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > self.maze_tol:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "RIGHT"
        return self._swing(direction, f"maze_wp{self.maze_wp_index}")

    def _farm_action(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Divert into a heart farm while low, then hand the hop back.

        Fail-soft on timeout / left-screen / ``max_farm_attempts``. Keep
        control through restock phases, not only ``FARM``.
        """
        if self._farm is not None:
            action = self._farm.step(snap)
            if self._farm.phase not in (HeartFarmPhase.DONE, HeartFarmPhase.FAILED):
                return action
            note = "farm_ok" if self._farm.success else "farm_gave_up"
            self.notes.append(f"{note}_{snap.filled_hearts}")
            self._farm = None
            self.stuck = 0
            return FrameAction(nes_idle_action(), note)
        if (
            self.farm_below_hearts <= 0
            or snap.level != 0
            or snap.filled_hearts >= self.farm_below_hearts
            or self.farm_attempts >= self.max_farm_attempts
            or worth_heart_farm(int(snap.screen)) is None
        ):
            return None
        self.farm_attempts += 1
        kwargs: dict[str, Any] = {
            "min_filled": self.farm_min_filled,
            "max_frames": self.farm_max_frames,
            "farm_screen": int(snap.screen),
            "waypoints": BAND_SWEEP_WAYPOINTS,
        }
        restock = self._restock_for(snap)
        if restock is not None:
            kwargs["restock_neighbor_screen"] = restock[0]
            kwargs["restock_direction"] = restock[1]
        self._farm = HeartFarmController(**kwargs)
        self.notes.append(f"farm_start_{snap.screen:02x}_{snap.filled_hearts}")
        return self._farm.step(snap)

    def _worth(self, screen: int) -> Any:
        """Catalog rupee-farm spot, skipping lynel/peahat/zora."""
        return worth_rupee_farm(int(screen))

    def _restock_for(self, snap: ZeldaSnapshot) -> tuple[int, str] | None:
        """Catalog restock, else current hop target/direction (leave toward hop)."""
        pair = restock_for(int(snap.screen))
        if pair is not None:
            return pair
        if self.hop_index < len(self.hops):
            hop = self.hops[self.hop_index]
            return (int(hop.target), hop.direction)
        return None

    def _rupee_farm_action(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Kill+restock on a farmable screen while short of ``need_rupees``.

        Fail-soft. Does not run after hops complete.
        """
        if self._rupee_farm is not None:
            action = self._rupee_farm.step(snap)
            phase = self._rupee_farm.phase
            if phase in (RupeeFarmPhase.FARM, RupeeFarmPhase.RETURN):
                return action
            if phase is RupeeFarmPhase.DONE:
                note = f"rupee_farm_ok_{snap.rupees}"
            else:
                note = "rupee_farm_gave_up"
            self.notes.append(note)
            self._rupee_farm = None
            self.stuck = 0
            return FrameAction(nes_idle_action(), note)
        if (
            self.need_rupees <= 0
            or snap.rupees >= self.need_rupees
            or snap.level != 0
            or snap.mode not in (PLAY_MODE, 8)
            or self.hop_index >= len(self.hops)
            or self.rupee_farm_attempts >= self.max_farm_attempts
            or self._worth(int(snap.screen)) is None
        ):
            return None
        hop = self.hops[self.hop_index]
        if self._in_maze_phase(snap, hop):
            return None
        restock = self._restock_for(snap)
        if restock is None or restock[1] not in _OPPOSITE:
            return None
        neighbor, direction = restock
        self.rupee_farm_attempts += 1
        self._rupee_farm = RupeeFarmController(
            target_rupees=self.need_rupees,
            farm_screen=int(snap.screen),
            restock_neighbor_screen=int(neighbor),
            restock_direction=direction,
            leftover_screen=int(snap.screen),
            max_frames=self.farm_max_frames,
        )
        self.notes.append(f"rupee_farm_start_{snap.screen:02x}")
        return self._rupee_farm.step(snap)

    def _rupee_scoop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """Walk onto a nearby drop. Hearts when not full; rupees/bombs when asked.

        Stays on-screen; refuses a drop on the hop's opposite edge. Contact
        pickup, no swing. ``need_rupees`` is the restock-farm target only.
        """
        if snap.mode != PLAY_MODE or snap.level != 0:
            return None
        heart = scoop_floor_drop(
            snap,
            types=HEART_FAIRY_DROP_TYPES,
            states=HEART_FAIRY_DROP_STATES,
            travel_dir=hop.direction,
            # A heart is worth crossing a screen for and a rupee is not, so
            # the heal reach is its own number. 0x7B dropped a heart *and* a
            # fairy on the pass that then died two screens later with
            # ``heal_hearts`` 0.0 (``pre_l1_beam4``): at 48 px both sat
            # outside the only layer that walks to a drop while travelling.
            radius=self.scoop_heal_radius,
            reason="scoop_heart",
            # Not ``filled_hearts < heart_containers``: that nibble is whole
            # hearts minus one, so it is true at full health and the walk
            # detours for a heart it cannot bank. ``combat.heal_wanted`` is
            # the honest test and it also catches the ``$0670`` chip that
            # takes the sword beam away.
            want=heal_wanted(snap),
        )
        if heart is not None:
            return heart
        rupee = scoop_floor_drop(
            snap,
            types=(RUPEE_DROP_OBJECT_TYPE,),
            states=RUPEE_DROP_STATES,
            travel_dir=hop.direction,
            radius=self.scoop_radius,
            reason="scoop_rupee",
            want=self.scoop_rupees or (
                self.need_rupees > 0 and snap.rupees < self.need_rupees
            ),
        )
        if rupee is not None:
            return rupee
        return scoop_floor_drop(
            snap,
            types=(BOMB_DROP_OBJECT_TYPE,),
            states=BOMB_DROP_STATES,
            travel_dir=hop.direction,
            radius=self.scoop_radius,
            reason="scoop_bomb",
            want=self.scoop_bombs and owns_bombs(snap) and int(snap.bombs) < 4,
        )

    def _occupancy_align_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """Walk to ``align_x`` on a walkable y; peel on miss. None on-column.

        East-mouth leftover (x≈240): LEFT toward the door column; occupancy
        miss → block cell → y-peel; no path → stand. Do not RIGHT-scroll.
        """
        if hop.direction not in ("UP", "DOWN") or hop.align_x is None or hop.y_band is not None:
            return None
        ax = int(hop.align_x)
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - ax) <= _ALIGN_X_TOL:
            self._hop_walker = None
            self._hop_walker_key = None
            return None
        # Inland off-column stays on align_and_push (credits-tape hops).
        if x < EDGE_EAST_X and x > EDGE_WEST_X:
            return None
        if self.stuck > self.stuck_threshold:
            return FrameAction(nes_idle_action(), f"hop{self.hop_index}_stand")

        key = (self.hop_index, int(snap.screen))
        if self._hop_walker is None or self._hop_walker_key != key:
            self._hop_walker = OccupancyWalker(grid=_ow_hop_grid(), slide=True)
            self._hop_walker_key = key
        walker = self._hop_walker
        xy = (x, y)
        direction = walker.next_dir(xy, (ax, y))
        # RIGHT at east mouth scrolls; LEFT at west edge leaves. UP/DOWN y-peel ok.
        forbidden: set[str] = set()
        if x >= EDGE_EAST_X:
            forbidden.add("RIGHT")
        if x <= EDGE_WEST_X:
            forbidden.add("LEFT")
        for _ in range(4):
            if direction is None or direction not in forbidden:
                break
            walker.grid.mark_blocked_ahead(*xy, direction)
            walker.path = None
            direction = walker.next_dir(xy)
        if direction is None or direction in forbidden:
            return FrameAction(nes_idle_action(), f"hop{self.hop_index}_stand")
        # Keep the occupancy lane; off-axis face would RIGHT-scroll the mouth.
        return swing_or_turn(
            self.phase_frames,
            direction,
            f"hop{self.hop_index}_occ",
            snap,
            period=self.swing_period,
            hold=self.swing_hold,
        )

    def _observe_threats(self, snap: ZeldaSnapshot) -> None:
        """Feed the tracker every frame so velocity is not always zero."""
        if self._tracker is None:
            self._tracker = ObjectTracker(shot_history=self.shot_history)
            self._evader = ReactiveEvader(bounds=_EVADE_BOUNDS)
        self._tracked = self._tracker.observe(snap)
        room = (int(snap.level), int(snap.screen))
        if room != self._evade_room:
            self._evade_room = room
            self._duck_walls = set()
            self._duck_xy = None
            self._duck_still = 0
            if self._evader is not None:
                self._evader.reset()  # slots reuse; drop a cross-scroll commit

    def _evade_blocked_dirs(self, snap: ZeldaSnapshot) -> set[str]:
        """Directions that would scroll Link off this screen (commit-wide)."""
        blocked: set[str] = set()
        if snap.link_y >= EDGE_SOUTH_Y - _EVADE_EDGE_MARGIN:
            blocked.add("DOWN")
        if snap.link_y <= EDGE_NORTH_Y + _EVADE_EDGE_MARGIN:
            blocked.add("UP")
        if snap.link_x >= EDGE_EAST_X - _EVADE_EDGE_MARGIN:
            blocked.add("RIGHT")
        if snap.link_x <= EDGE_WEST_X + _EVADE_EDGE_MARGIN:
            blocked.add("LEFT")
        return blocked

    def _evade_goal(
        self, snap: ZeldaSnapshot, hop: ScreenHop | None
    ) -> tuple[int, int] | None:
        """Tie-break escapes toward the hop's lane, not away from it."""
        if hop is None:
            return None
        if hop.align_x is not None:
            return (int(hop.align_x), int(snap.link_y))
        if hop.align_y is not None:
            return (int(snap.link_x), int(hop.align_y))
        if hop.y_band is not None:
            lo, hi = hop.y_band
            return (int(snap.link_x), (int(lo) + int(hi)) // 2)
        return None

    def _shot_first(self, snap: ZeldaSnapshot) -> bool:
        """True when an unblockable shot lands inside the dodge window.

        Assessed against the *shots alone*, never against ``assess`` over
        every hazard: a body close enough to be in the blade box is by
        construction close enough to own ``Impact.source``, so asking the
        combined impact whether the next thing to land is a shot answers
        "no" on exactly the frames this rule exists for.

        Zora spit (``0x55``) is the live case — unkillable, and
        ``behaviors.shield_blocks`` says the small shield does not stop it —
        so no swing shortens that contact, and ``dodgeable`` is the other
        half: a shot already inside the pad is a hit, not a decision.
        """
        shots = tuple(
            t
            for t in self._tracked
            if t.hazard is HazardClass.PROJECTILE and not t.blockable
        )
        if not shots:
            return False
        impact = assess(
            (int(snap.link_x), int(snap.link_y)), shots, bounds=_EVADE_BOUNDS
        )
        return impact.within(_SHOT_OVER_SWORD_TTC) and dodgeable(impact)

    def _body_first(self, snap: ZeldaSnapshot, shot_gap: int) -> bool:
        """True when a body is too close for the shot to own the frame.

        The duck walks, and a walk through a leever swarm is a contact. Live
        0x7B (``zhit3`` f=4691): six frames of ``spit_duck`` with a leever
        closing 12 px to 8 took the walk's first hit — at full health, which
        is the hit that costs the sword beam (``$0670`` chips it off) and
        turns every screen after it into melee. Inside this pad the honest
        answers are the blade and the peel, and both live below this rung.
        ``ShotPolicy.face`` keeps the same rule for the shield.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        for track in self._tracked:
            if track.hazard is not HazardClass.BODY:
                continue
            gap = chebyshev(lx, ly, int(track.x), int(track.y))
            if gap <= _SPIT_DUCK_BODY_PAD and gap <= int(shot_gap):
                return True
        return False

    def _note_duck_wall(self, snap: ZeldaSnapshot, direction: str) -> None:
        note = f"duck_wall_{int(snap.screen):02x}_{direction.lower()}"
        if note not in self.notes:
            self.notes.append(note)

    def _spit_duck(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Step off a closing unblockable shot, ahead of every other rung.

        This is the Zora rule, and it sits above the sword on purpose. The
        evader below it yields the frame whenever the blade reaches a body
        (``evade_yield_to_sword`` owned 687 of 6384 walk frames in
        ``zhit1``), which on 0x7B / 0x7C is almost continuous — those screens
        hold six leevers *and* the Zora that shares them, so the one rung
        that could answer the spit was handed off for exactly the window it
        was written for. Nothing below this line can answer a ``0x55``: it
        cannot be killed by the walk (``prey.SKIP_TYPES``), the small shield
        does not block it, and ``assess`` scores a muzzle that has not
        launched yet as safe because it is not moving.

        Closing only. A shot going away is geometry, not a threat, and
        ducking it would spend the hop's frames on nothing. :meth:`_body_first`
        is the other side: the duck is a *walk*, so a body already inside its
        own dodge pad outranks the shot.
        """
        if not self.evade:
            return None
        nodes = self._geo_walkable(snap) if self.shot_model else None
        if nodes:
            return self._spit_escape(snap, nodes)
        lx, ly = int(snap.link_x), int(snap.link_y)
        best: TrackedObject | None = None
        best_gap = 10**9
        for track in self._tracked:
            if track.hazard is not HazardClass.PROJECTILE or track.blockable:
                continue
            if not track.closing_on(lx, ly):
                continue
            gap = chebyshev(lx, ly, int(track.x), int(track.y))
            if gap <= _SPIT_DUCK_RADIUS and gap < best_gap:
                best, best_gap = track, gap
        if best is None or self._body_first(snap, best_gap):
            self._duck_dir = self._duck_slot = None
            self._duck_frames = 0
            self._duck_xy = None
            self._duck_still = 0
            return None
        # A duck that does not move Link is not a duck. ``_EVADE_BOUNDS`` is
        # the scroll rectangle and knows nothing about the coast's rocks, so
        # the only honest test is the one the frame just ran: live 0x7B
        # (``zhit2`` f=4684) held UP for eight frames at (48, 133) against a
        # wall while a leever closed from 12 px to 8 and took the walk's
        # first hit — at full health, with the streak on 10, which is the
        # single most expensive frame on the corridor (it costs the beam and
        # the forced 5-rupee at once). The threshold is
        # ``_SPIT_DUCK_STILL_CAP`` rather than one frame because Link's
        # 1 px/frame walk genuinely repeats a pixel now and then.
        here = (int(snap.link_x), int(snap.link_y))
        cell = (here[0] // _DUCK_CELL, here[1] // _DUCK_CELL)
        if self._duck_dir is not None and not link_busy(snap):
            # ``link_busy`` is the other reason Link does not move: the ROM
            # pins him for the whole sword animation, and charging those
            # frames to the terrain is how ``zfixB`` decided both sides of
            # 0x7B were rock.
            if self._duck_xy == here:
                self._duck_still += 1
                if self._duck_still >= _SPIT_DUCK_STILL_CAP:
                    self._duck_walls.add((self._duck_dir, cell[0], cell[1]))
                    self._note_duck_wall(snap, self._duck_dir)
                    self._duck_dir = None
                    self._duck_still = 0
            else:
                self._duck_still = 0
        self._duck_xy = here
        slot = int(best.slot)
        step: str | None = None
        bodies = overworld_threat_objects(snap)
        walls = {
            direction
            for direction in ("UP", "DOWN", "LEFT", "RIGHT")
            if (direction, cell[0], cell[1]) in self._duck_walls
        }
        if (
            self._duck_dir is not None
            and self._duck_slot == slot
            and self._duck_frames < _SPIT_DUCK_COMMIT
            and box_step(lx, ly, self._duck_dir, _EVADE_BOUNDS) is not None
            # The commit is a commit against the *shot*, never against a
            # body: ``zfixA`` walked 0x7B's leevers down four times, and a
            # dodge that steps into a leever has traded a 0.5 heart shot for
            # a 0.5 heart contact plus the streak.
            and not _step_into_body(lx, ly, self._duck_dir, bodies)
        ):
            step = self._duck_dir
        else:
            step = perpendicular(
                lx, ly, int(best.x), int(best.y), _EVADE_BOUNDS, bodies,
                avoid=walls,
            )
            self._duck_frames = 0
        band = None if self._threat_hop is None else self._threat_hop.y_band
        step = keep_y_band(
            step, lx, ly, int(best.x), int(best.y), _EVADE_BOUNDS, bodies,
            band=band, avoid=walls,
        )
        if step is None:
            # Boxed in on both sides of the bearing. The push is the honest
            # answer; handing back a direction that closes the gap is how the
            # old ``answer_projectile`` wall-flip took the 0x7C hit.
            self._duck_dir = self._duck_slot = None
            return None
        self._duck_frames = self._duck_frames + 1 if step == self._duck_dir else 1
        self._duck_dir, self._duck_slot = step, slot
        self.spit_ducks += 1
        self.evade_reasons["spit_duck"] = self.evade_reasons.get("spit_duck", 0) + 1
        return FrameAction(nes_action(step), "spit_duck")

    def _spit_escape(
        self, snap: ZeldaSnapshot, nodes: frozenset[tuple[int, int]]
    ) -> FrameAction | None:
        """The duck against the shot's own line, on the measured lattice.

        ``perpendicular`` crosses the *bearing's* major axis, and when that
        side is rock it falls back to whatever is left, which on a diagonal
        shot is the shot's own heading: six of the seven 0x7B/0x7C spit hits
        in ``pre_l1`` trace ``z1`` were Link walking with the fireball until
        he stopped and it caught him. This scores every held input against
        each shot's straight flight (``common.shot_escape``) and only takes
        the frame when some input would be hit. A shot still on the muzzle
        (``ZORA_MUZZLE_DWELL``, zero velocity) is flown at Link from now.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        shots: list[tuple[float, float, float, float, int]] = []
        nearest = 10**9
        for track in self._tracked:
            if track.hazard is not HazardClass.PROJECTILE:
                continue
            if track.blockable and not self.shot_model_rocks:
                continue
            tx, ty = int(track.x), int(track.y)
            gap = chebyshev(lx, ly, tx, ty)
            if gap > _SPIT_DUCK_RADIUS:
                continue
            vx, vy = float(track.vx), float(track.vy)
            if max(abs(vx), abs(vy)) < SHOT_DWELL_SPEED:
                ax, ay = lx + 8 - (tx + 4), ly - ty
                norm = max(1.0, (ax * ax + ay * ay) ** 0.5)
                vx, vy = ZORA_SHOT_SPEED * ax / norm, ZORA_SHOT_SPEED * ay / norm
            elif not track.closing_on(lx, ly):
                continue
            shots.append((float(tx), float(ty), vx, vy, 4))
            nearest = min(nearest, gap)
        if not shots or self._body_first(snap, nearest):
            self._duck_dir = self._duck_slot = None
            self._duck_frames = 0
            return None
        prefer: list[str] = []
        if self._duck_dir is not None and self._duck_frames < _SPIT_DUCK_COMMIT:
            prefer.append(self._duck_dir)
        hop = self._threat_hop
        if hop is not None and hop.direction not in prefer:
            prefer.append(hop.direction)
        direction, needed = shot_escape(
            lx, ly, shots, _EVADE_BOUNDS,
            nodes=nodes, bodies=overworld_threat_objects(snap), prefer=tuple(prefer),
        )
        if not needed:
            self._duck_dir = self._duck_slot = None
            self._duck_frames = 0
            return None
        self._duck_frames = self._duck_frames + 1 if direction == self._duck_dir else 1
        self._duck_dir = direction
        self.spit_ducks += 1
        self.evade_reasons["spit_duck"] = self.evade_reasons.get("spit_duck", 0) + 1
        if direction is None:
            return FrameAction(nes_idle_action(), "spit_stand")
        return FrameAction(nes_action(direction), "spit_duck")

    def attach_rollout(self, env: Any, **kwargs: Any) -> RolloutEvader:
        """Bind the ROM-truth evader to the live env and select that arm.

        The seam a pin uses to pick which evader is under test. ``env`` is
        the emulator the walk is already running on -- **one emulator per
        process** (``AGENTS.md``), so this never makes one, and
        :class:`~zelda_i.rollout.Rollout` restores it in a ``finally`` on
        every fan. ``kwargs`` are the budget knobs
        (``replan_frames`` / ``trigger_radius`` / ``screen_budget`` / ...),
        passed to the constructor rather than set on the instance afterwards:
        a ``@dataclass`` copies its defaults into ``__init__`` when the class
        is created, so ``setattr`` on the *class* is a no-op on every
        instance and silently measures the unablated arm (``AGENTS.md``).

        Returns the evader so a probe can read its ledger directly.
        """
        self._rollout = RolloutEvader(Rollout(env), **kwargs)
        self.rollout_evade = True
        return self._rollout

    def _rung_rollout_evade(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """The ROM-truth dodge, when it is bound and has something to say.

        Off by default: with no ``attach_rollout`` this declines on every
        frame and the ladder is the chain that was here before.

        It does *not* inherit either of the model's gates. ``dodgeable`` is
        ``assess``, which is the thing under test, and the sword yield hides
        the Zora muzzle for the same reason: while the shot is still, neither
        ``_shot_first`` nor the straight-line tracker can see the coming hit.
        Standing is already one of the rollout candidates. If standing is as
        safe as walking, ``no_gain`` declines to the reactive rung/hunter; if
        a walk survives and standing is hit, the ROM has measured why the
        dodge outranks the swing.
        """
        if not self.rollout_evade or self._rollout is None:
            return None
        step = self._rollout.decide(
            snap, self._tracked, blocked=self._evade_blocked_dirs(snap)
        )
        if step is None or step.direction is None:
            return None
        self.evade_reasons[step.reason] = self.evade_reasons.get(step.reason, 0) + 1
        self.evades += 1
        # Do not slash-walk: A frames stop Link inside the pad.
        return FrameAction(nes_action(step.direction), step.reason)

    def threat_rungs(self) -> tuple[Rung, ...]:
        """The threat ladder, as data. All three rungs, always registered.

        The rollout rung stays on the ladder even with nothing bound, so the
        two arms are the *same* ladder with one flag between them and a
        census taken on either is directly comparable -- ``threat_rollout: 0``
        is a reading, not a missing row.
        """
        return (
            Rung("threat_duck", THREAT_RUNG_DUCK, self._spit_duck),
            Rung("threat_rollout", THREAT_RUNG_ROLLOUT, self._rung_rollout_evade),
            Rung("threat_reactive", THREAT_RUNG_REACTIVE, self._rung_reactive_evade),
        )

    @property
    def threat_arbiter(self) -> Arbiter:
        """The threat ladder. Built on first use, dropped by ``reset``."""
        if self._threat_arbiter is None:
            self._threat_arbiter = Arbiter(self.threat_rungs())
        return self._threat_arbiter

    def _threat_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop | None
    ) -> FrameAction | None:
        """Reactive first, before any hop rule answers this frame.

        Yields ``None`` when standing is safe so the hop drives quiet frames.
        The guards here are a prologue, not rungs: they say whether a threat
        decision is meaningful at all, which is not a claim on the frame.
        """
        if not self.evade or self._evader is None:
            return None
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return None
        self._threat_hop = hop
        return self.threat_arbiter.decide(snap)

    def _rung_reactive_evade(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """``threat.ReactiveEvader`` and the gates it runs behind.

        Unchanged policy -- this is the tail of the old ``_threat_action``
        chain, lifted onto the ladder so the rollout rung has somewhere to
        sit *above* it and so the census can price the two arms apart.
        """
        if self._evader is None:
            return None
        hop = self._threat_hop
        hazards = tuple(t for t in self._tracked if t.is_hazard)
        stand = assess(
            (int(snap.link_x), int(snap.link_y)), hazards, bounds=_EVADE_BOUNDS
        )
        shot_first = self._shot_first(snap)
        if self.hunter is not None and self.hunter.striking(snap):
            # Blade already reaches; hunt owns this frame — unless what is
            # about to land is a *shot*. ``evade_yield_to_sword`` was the
            # largest single evade reason on the coast walk (383 frames of
            # 6552, ``pre_l1_beam4``) and the leever screens 0x7B/0x7C keep a
            # body in the blade box almost continuously, so the Zora sharing
            # those screens fired into an evader that had been handed off for
            # the whole window. The sword is not an answer to 0x55: it is not
            # killable and the small shield does not block it, so trading the
            # dodge for the swing spends a heart to save a frame.
            if not shot_first:
                self.evade_reasons["evade_yield_to_sword"] = (
                    self.evade_reasons.get("evade_yield_to_sword", 0) + 1
                )
                self._evader.reset()
                return None
            self.evade_reasons["evade_shot_over_sword"] = (
                self.evade_reasons.get("evade_shot_over_sword", 0) + 1
            )
        if not dodgeable(stand) and not shot_first and self.stuck < _WEDGE_STUCK_FRAMES:
            # In-pad while hop still moves: yield; ``stuck`` is the strict counter.
            self.evade_reasons["evade_in_pad"] = (
                self.evade_reasons.get("evade_in_pad", 0) + 1
            )
            self._evader.reset()
            return None
        blocked = self._evade_blocked_dirs(snap)
        decision = self._evader.decide(
            snap,
            self._tracked,
            goal=self._evade_goal(snap, hop),
            blocked_dirs=blocked,
        )
        if decision is not None:
            reason = decision.reason
            self.evade_reasons[reason] = self.evade_reasons.get(reason, 0) + 1
            if decision.direction is not None:
                parry = self._parry(snap, decision.source_slot)
                if parry is not None:
                    return parry
                self.evades += 1
                # Do not slash-walk: A frames stop Link inside the pad.
                return FrameAction(nes_action(decision.direction), f"evade_{reason}")
            if decision.shield:
                return FrameAction(nes_idle_action(), f"evade_{reason}")
            # ``evade_no_gain`` / ``evade_boxed_in``: hop still slashes.
        return None

    def _parry(self, snap: ZeldaSnapshot, slot: int | None) -> FrameAction | None:
        """Sword the inbound body when the blade already reaches it.

        Evader reasons about feet only. ``slot`` keeps the sword on *the*
        inbound threat.
        """
        if slot is None:
            return None
        target = next(
            (
                obj
                for obj in overworld_threat_objects(snap)
                if int(obj.slot) == int(slot)
            ),
            None,
        )
        if target is None or is_projectile(target):
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        dx, dy = int(target.x) - lx, int(target.y) - ly
        if abs(dx) >= abs(dy):
            face = "RIGHT" if dx > 0 else "LEFT"
        else:
            face = "DOWN" if dy > 0 else "UP"
        # The blade's near end (``hunt.blade_lands``): a body inside it cannot
        # be cut, so a parry there is 13 pinned frames touching it. And one
        # slot gets ``_PARRY_SLOT_MAX_FRAMES``: 0x7D's octoroks held
        # ``evade_parry``/``parry_recover`` for 3047 of a 4991 frame walk.
        if not blade_lands(lx, ly, face, int(target.x), int(target.y)):
            return None
        key = (int(snap.screen), int(slot), int(target.type_id))
        if key != self._parry_key:
            self._parry_key, self._parry_frames = key, 0
        self._parry_frames += 1
        if self._parry_frames > _PARRY_SLOT_MAX_FRAMES:
            return None
        self.parries += 1
        if int(snap.facing) != direction_to_facing(face):
            return FrameAction(nes_action(face), "evade_parry_face")  # no turn in place
        if self.phase_frames % self.swing_period < self.swing_hold:
            return FrameAction(nes_action("A"), "evade_parry")
        return FrameAction(nes_idle_action(), "evade_parry_recover")

    def _lane_no_gain(self, snap: ZeldaSnapshot, travel: str, lx: int) -> bool:
        """True once the peel has stopped buying travel on this screen.

        Ticks only on frames this branch is actually *reached* — it sits under
        the hunt in ``_do_hop``, so a fight holding Link still is not charged
        here. What is charged is a lane frame that leaves Link no closer to the
        exit than the best pixel he has already stood on, which is the whole
        shape of the 0x7C tug-of-war: the peel steps LEFT, the plain push steps
        RIGHT, and x reads 24, 25, 24, 25 for 3593 frames.

        The latch is per ``(hop_index, screen)`` on purpose. Letting the branch
        back in on the first pixel the push wins would just restart the
        stand-off one pixel east — 215 px of screen at one cap per pixel is
        slower than the 4000-frame ``_grinding`` floor it is meant to beat.
        """
        key = (int(self.hop_index), int(snap.screen))
        if key != self._lane_key:
            self._lane_key = key
            self._lane_gain = lx
            self._lane_nogain = 0
            self._lane_off = False
        if self._lane_off:
            return True
        best = self._lane_gain if self._lane_gain is not None else lx
        gained = lx > best if travel == "RIGHT" else lx < best
        if gained:
            self._lane_gain = lx
            self._lane_nogain = 0
            return False
        self._lane_nogain += 1
        if self._lane_nogain > _OCCUPIED_LANE_NO_GAIN_CAP:
            self._lane_off = True
            self.notes.append(f"lane_nogain_{self.hop_index}_{int(snap.screen):02x}")
            return True
        return False

    def _occupied_lane_action(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        play = snap.level == 0 and snap.mode == PLAY_MODE and not snap.transitioning
        if not self.occupied_lane or not play:
            return None
        hazards = tuple(t for t in self._tracked if t.is_hazard)
        lx, ly = int(snap.link_x), int(snap.link_y)
        if not hazards or not dodgeable(assess((lx, ly), hazards, bounds=_EVADE_BOUNDS)):
            self._lane_stand = 0
            self._lane_steer = 0
            return None
        travel = hop.direction
        # RIGHT/LEFT only; DOWN peels stay off until a sitting owns that column.
        if travel not in ("LEFT", "RIGHT"):
            self._lane_stand = 0
            self._lane_steer = 0
            return None
        if self._lane_no_gain(snap, travel, lx):
            self._lane_stand = 0
            self._lane_steer = 0
            return None
        iy = int(hop.align_y) if hop.align_y is not None else ly
        blocked = _lane_blocked(lx, iy, travel, hazards)
        if not blocked:
            self._lane_stand = 0
            self._lane_steer = 0
            return None
        reason = f"hop{self.hop_index}_lane"
        # Parallel to the hop lane (not Link's row); ``toward`` picks the side.
        goal = _parallel_lane(
            lx, iy, travel, hazards, self._evade_blocked_dirs(snap), toward=(lx, ly),
        )
        step = None
        if goal is not None:
            gx, gy = goal
            if gy != ly:
                step = "DOWN" if gy > ly else "UP"
            elif gx != lx:
                step = "RIGHT" if gx > lx else "LEFT"
            else:
                step = travel
            dx, dy = WALK_DELTA[step]
            if step != travel and _pad_hits(lx + dx, ly + dy, hazards):
                step = None  # the peel itself would walk into a body
        if step is not None:
            self._lane_stand = 0
            self._lane_steer = 0 if step == travel else self._lane_steer + 1
            if self._lane_steer <= _OCCUPIED_LANE_STEER_CAP:
                return FrameAction(nes_action(step), reason)
        # Own row if clear, then stand, then yield.
        if not _lane_blocked(lx, ly, travel, hazards):
            self._lane_stand = 0
            return FrameAction(nes_action(travel), f"{reason}_row")
        self._lane_stand += 1
        if self._lane_stand > _OCCUPIED_LANE_STAND_CAP:
            self._lane_stand = 0
            self._lane_steer = 0
            return None
        return FrameAction(nes_idle_action(), f"{reason}_stand")

    def _stall_escape(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """Committed route out of a pocket toward this hop's exit.

        Starts only past ``stuck_threshold``. Hunt-only: the hunter holds
        the learned grid. Holds until scroll, no route, or commit expiry.
        """
        if self.hunter is None:
            return None
        screen = int(snap.screen)
        if self._escape_frames <= 0:
            if self.stuck <= self.stuck_threshold:
                return None
            self._escape_frames = self.escape_commit_frames
            self._escape_screen = screen
            self.stall_escapes += 1
            self.notes.append(f"stall_escape_{screen:02x}_{snap.link_x}_{snap.link_y}")
        elif screen != self._escape_screen:
            self._escape_frames = 0
            return None
        self._escape_frames -= 1
        act = self.hunter.escape_for(
            snap, self.frames, hop, f"hop{self.hop_index}_escape"
        )
        if act is None:
            self._escape_frames = 0
        return act

    def _hunt_action(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """Clear this screen before crossing it. After ``recover_off_edge``."""
        if self.hunter is None:
            return None
        if on_arrival_edge(hop.direction, snap):
            return None
        return self.hunter.step(
            snap, self.frames, lane=hop_lane(hop), y_band=hop.y_band
        )

    def _grinding(self, snap: ZeldaSnapshot) -> bool:
        """True once this hop has spent its budget on this screen.

        The optional rungs each have their own local cap and none of them
        compose: on 0x7C the occupied-lane steer and the plain push alternated
        one frame each for 24914 frames (``pre_l1_wedge1``: ``hop`` 12459,
        ``hop_lane`` 12455) because every steer reset the steer counter the
        moment it chose the travel direction, and ``track_stuck`` saw a Link
        who was moving. Only the frames actually spent on this screen, under
        this hop, are proof against that — and the answer to a hop that has
        spent them is to stop being clever and push.
        """
        key = (int(self.hop_index), int(snap.screen))
        if key != self._hop_screen:
            self._hop_screen = key
            self._hop_screen_frames = 0
        self._hop_screen_frames += 1
        if self._hop_screen_frames == self.hop_screen_max_frames + 1:
            self.notes.append(f"hop_grind_{self.hop_index}_{int(snap.screen):02x}")
        return self._hop_screen_frames > self.hop_screen_max_frames

    # ---------------------------------------------------- hop rungs ---
    # One method per branch of the old chain, each with the ``RungFn``
    # shape. They read ``self._hop`` / ``self._hop_grinding``, which
    # ``_do_hop`` sets once a frame before it walks the ladder.

    def _rung_extra(self, snap: ZeldaSnapshot) -> FrameAction | None:
        return self._extra_hop_action(snap, self._hop)

    def _rung_maze(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if not self._in_maze_phase(snap, self._hop):
            return None
        return self._follow_maze(snap)

    def _rung_scoop(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self._hop_grinding:
            return None
        return self._rupee_scoop(snap, self._hop)

    def _rung_occupancy(self, snap: ZeldaSnapshot) -> FrameAction | None:
        return self._occupancy_align_action(snap, self._hop)

    def _rung_stall_escape(self, snap: ZeldaSnapshot) -> FrameAction | None:
        return self._stall_escape(snap, self._hop)

    def _rung_unstick(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.stuck <= self.stuck_threshold:
            return None
        action, self.stuck = unstick_wiggle(self.stuck)
        return action

    def _rung_edge(self, snap: ZeldaSnapshot) -> FrameAction | None:
        return recover_off_edge(snap, self._hop.direction, swing=self._swing)

    def _rung_hunt(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self._hop_grinding:
            return None
        return self._hunt_action(snap, self._hop)

    def _rung_lane(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self._hop_grinding:
            return None
        return self._occupied_lane_action(snap, self._hop)

    def bind_env(self, env: Any) -> None:
        """Keep the env so ``_rung_geo`` can read the room tile map. Read only."""
        self._env = env

    def _geo_walkable(self, snap: ZeldaSnapshot) -> frozenset[tuple[int, int]] | None:
        """This screen's lattice, re-read every 64 frames (a secret opens tiles)."""
        if self._env is None or not self.geo:
            return None
        key = (int(snap.screen), self.frames // 64)
        if self._geo_key != key:
            from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes

            ram = self._env.get_ram()
            self._geo_nodes = ow_walkable_nodes(ram) if has_room_tile_map(ram) else None
            self._geo_key = key
        return self._geo_nodes

    @classmethod
    def _geo_goals(cls, hop: ScreenHop, nodes: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
        """Edge nodes that scroll ``hop.direction`` and satisfy its align.

        A hand-measured align can name a row the tile map says is rock at
        the edge: 0x79's ``SCREEN_79_BEACH_Y`` 165 dead-ends at x=192 (y 133
        and 141 scroll, measured by walking them). Then the nearest walkable
        edge node to the align is the goal, not "no goal".
        """
        goals = cls._geo_goals_strict(hop, nodes)
        if goals:
            return goals
        edge = cls._geo_goals_strict(
            ScreenHop(hop.target, hop.direction), nodes
        )
        if not edge:
            return set()
        if hop.direction in ("UP", "DOWN") and hop.align_x is not None:
            want, axis = int(hop.align_x), 0
        elif hop.direction in ("LEFT", "RIGHT") and hop.y_band is not None:
            want, axis = (int(hop.y_band[0]) + int(hop.y_band[1])) // 2, 1
        elif hop.direction in ("LEFT", "RIGHT") and hop.align_y is not None:
            want, axis = int(hop.align_y), 1
        else:
            return edge
        gap = min(abs(c[axis] - want) for c in edge)
        return {c for c in edge if abs(c[axis] - want) == gap}

    @staticmethod
    def _geo_goals_strict(hop: ScreenHop, nodes: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
        xs = sorted({x for x, _ in nodes})
        ys = sorted({y for _, y in nodes})
        if not xs or not ys:
            return set()
        if hop.direction in ("UP", "DOWN"):
            y = ys[0] if hop.direction == "UP" else ys[-1]
            cand = {(x, y) for x in xs}
            if hop.align_x is not None:
                cand = {c for c in cand if abs(c[0] - int(hop.align_x)) <= 5}
        else:
            x = xs[0] if hop.direction == "LEFT" else xs[-1]
            cand = {(x, y) for y in ys}
            if hop.y_band is not None:
                lo, hi = hop.y_band
                cand = {c for c in cand if int(lo) <= c[1] <= int(hi)}
            elif hop.align_y is not None:
                cand = {c for c in cand if abs(c[1] - int(hop.align_y)) <= 5}
        return cand & set(nodes)

    @staticmethod
    def _geo_direct_clear(
        hop: ScreenHop,
        nodes: frozenset[tuple[int, int]],
        start: tuple[int, int],
        goals: set[tuple[int, int]],
    ) -> bool:
        """True when the align-then-push the fall-through walks is all floor."""
        sx, sy = start
        if hop.direction in ("UP", "DOWN"):
            gx = sx if hop.align_x is None or abs(sx - int(hop.align_x)) <= 5 else min(
                (g[0] for g in goals), key=lambda gx_: abs(gx_ - int(hop.align_x))
            )
            gy = next(iter(goals))[1]
            leg1 = [(x, sy) for x in range(min(sx, gx), max(sx, gx) + 1, LATTICE_STEP)]
            leg2 = [(gx, y) for y in range(min(sy, gy), max(sy, gy) + 1, LATTICE_STEP)]
        else:
            gy_set = sorted(g[1] for g in goals)
            gy = sy if sy in gy_set else min(gy_set, key=lambda v: abs(v - sy))
            gx = next(iter(goals))[0]
            leg1 = [(sx, y) for y in range(min(sy, gy), max(sy, gy) + 1, LATTICE_STEP)]
            leg2 = [(x, gy) for x in range(min(sx, gx), max(sx, gx) + 1, LATTICE_STEP)]
        return all(n in nodes for n in leg1 + leg2)

    def _rung_geo(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Walk the measured route to the hop's edge when the straight one is rock.

        Declines whenever the align-then-push is all floor, so a screen the
        blind walk already crosses is walked exactly as before.
        """
        if self._hop_grinding or snap.level != 0 or snap.mode != PLAY_MODE:
            return None
        if snap.transitioning:
            return None
        nodes = self._geo_walkable(snap)
        if not nodes:
            return None
        hop = self._hop
        goals = self._geo_goals(hop, nodes)
        if not goals:
            return None
        x, y = int(snap.link_x), int(snap.link_y)
        starts = [n for n in lattice_starts(x, y) if n in nodes]
        if not starts:
            return None
        start = min(starts, key=lambda n: abs(n[0] - x) + abs(n[1] - y))
        key = (self.hop_index, int(snap.screen), len(goals))
        route = self._geo_route if self._geo_route_key == key else None
        # The fall-through walks the hop's own align; if that has no edge node
        # it cannot finish, so only a reachable align earns the decline.
        strict = self._geo_goals_strict(hop, nodes)
        if route is None and strict and self._geo_direct_clear(hop, nodes, start, strict):
            self._geo_route = self._geo_route_key = None
            return None
        if route:
            # Drop corners already reached; replan when knocked off the leg.
            while route and (x, y) == route[0]:
                route = route[1:]
            if route and x != route[0][0] and y != route[0][1]:
                route = None
        if not route:
            route = lattice_route(nodes, (x, y), goals)
            if route is None:
                self._geo_route = self._geo_route_key = None
                return None
            if not route:
                self._geo_route = self._geo_route_key = None
                return None
        self._geo_route, self._geo_route_key = route, key
        direction = lattice_step(x, y, route[0])
        if direction is None:
            return None
        self.geo_frames += 1
        return self._swing(direction, f"hop{self.hop_index}_geo")

    def hop_rungs(self) -> tuple[Rung, ...]:
        """This controller's hop ladder, as data.

        Registration order is irrelevant — :class:`~zelda_i.overworld.arbiter.Arbiter`
        sorts on ``(priority, name)`` — so a subclass adds a rung by
        appending to this tuple and *placing* it, never by moving a line.

        The beam is the hunter's own bound ``take_beam``: the hop no longer
        reaches into the hunter's blade to get the shot high enough in the
        chain, it declares ``HOP_RUNG_BEAM`` and the arbiter does the rest.
        A hunterless path has no beam rung at all, because there is no bound
        method to register — ``hop_hunt`` stays on the ladder and declines,
        which is what ``_hunt_action`` already did.
        """
        rungs = [
            Rung("hop_extra", self.extra_hop_priority, self._rung_extra),
            Rung("hop_maze", HOP_RUNG_MAZE, self._rung_maze),
            Rung("hop_scoop", HOP_RUNG_SCOOP, self._rung_scoop),
            Rung("hop_occupancy", HOP_RUNG_OCCUPANCY, self._rung_occupancy),
            Rung("hop_stall_escape", HOP_RUNG_STALL_ESCAPE, self._rung_stall_escape),
            Rung("hop_unstick", HOP_RUNG_UNSTICK, self._rung_unstick),
            Rung("hop_edge", HOP_RUNG_EDGE, self._rung_edge),
            Rung("hop_hunt", HOP_RUNG_HUNT, self._rung_hunt),
            Rung("hop_geo", HOP_RUNG_GEO, self._rung_geo),
            Rung("hop_lane", HOP_RUNG_LANE, self._rung_lane),
        ]
        if self.hunter is not None:
            rungs.append(Rung("hop_beam", HOP_RUNG_BEAM, self.hunter.take_beam))
        return tuple(rungs)

    @property
    def hop_arbiter(self) -> Arbiter:
        """The hop ladder. Built on first use, rebuilt on a hunter swap."""
        if self._hop_arbiter is None or self._hop_arbiter_hunter is not self.hunter:
            self._hop_arbiter = Arbiter(self.hop_rungs())
            self._hop_arbiter_hunter = self.hunter
        return self._hop_arbiter

    def _do_hop(self, snap: ZeldaSnapshot) -> FrameAction:
        hop = self.hops[self.hop_index]
        # Prologue, not rungs. ``_advance_hop`` decides which hop the frame
        # even belongs to, and ``_grinding`` is a per-frame side effect whose
        # verdict the rungs below read — neither is a claim on the frame.
        advanced = self._advance_hop(snap, hop)
        if advanced is not None:
            return advanced
        self._hop = hop
        self._hop_grinding = self._grinding(snap)
        act = self.hop_arbiter.decide(snap)
        if act is not None:
            return act
        return align_and_push(
            snap,
            direction=hop.direction,
            reason=f"hop{self.hop_index}",
            align_x=hop.align_x,
            align_y=hop.align_y,
            y_band=hop.y_band,
            stuck=0,
            stuck_threshold=self.stuck_threshold,
            swing=self._swing,
            align_x_at_wall=hop.align_x_at_wall,
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Decide the frame, then record which rung decided it.

        The census is the only way to answer "where did 1759 frames on one
        transit screen go": the ladder is source-line order (see
        ``overworld.arbiter``), so a run report can otherwise name the
        *hop index* it stalled on and nothing about the behaviour that owned
        the frames. Counting is free and changes no decision.
        """
        act = self._decide(snap)
        screen = f"{int(snap.screen):#04x}"
        by_screen = self.reason_by_screen.setdefault(screen, {})
        reason = _reason_key(act.reason)
        by_screen[reason] = by_screen.get(reason, 0) + 1
        return act

    def _decide(self, snap: ZeldaSnapshot) -> FrameAction:
        self._nav_snap = snap
        if self.evade or self.occupied_lane:
            self._observe_threats(snap)
        if self.hunter is not None:
            # Census every frame: kills land during evades/farms/hop swings too.
            self.hunter.observe(snap)
        self.frames += 1
        self.phase_frames += 1
        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )
        self.hits_taken, self.last_health, self.stuck = track_knockback(
            snap,
            last_health=self.last_health,
            hits=self.hits_taken,
            stuck=self.stuck,
        )
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.mode == 17:
            return self._fail("link_death")
        if self._at_stop(snap):
            return self._finish("path_stop")
        early = self._before_play(snap)
        if early is not None:
            return early
        # Active farms own their own scroll/mode handling (restock leave/return).
        farm_busy = self._farm is not None or self._rupee_farm is not None
        if snap.transitioning and not farm_busy:
            return self._handle_transition(snap)
        if snap.mode not in self.allowed_modes and not farm_busy:
            return wake_or_wait_mode(self.phase_frames, snap.mode)
        farm = self._farm_action(snap)
        if farm is not None:
            return farm
        rupee = self._rupee_farm_action(snap)
        if rupee is not None:
            return rupee
        hop = self.hops[self.hop_index] if self.hop_index < len(self.hops) else None
        threat = self._threat_action(snap, hop)
        if threat is not None:
            return threat
        if hop is None:
            return self._after_hops(snap)
        return self._do_hop(snap)


__all__ = [
    "DEFAULT_SWING_PERIOD",
    "DEFAULT_SWING_HOLD",
    "DEFAULT_STUCK_THRESHOLD",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_HOP_SCREEN_MAX_FRAMES",
    "DEFAULT_SCOOP_RADIUS",
    "DEFAULT_SCOOP_HEAL_RADIUS",
    "HOP_RUNG_EXTRA",
    "HOP_RUNG_MAZE",
    "HOP_RUNG_BEAM",
    "HOP_RUNG_SCOOP",
    "HOP_RUNG_OCCUPANCY",
    "HOP_RUNG_STALL_ESCAPE",
    "HOP_RUNG_UNSTICK",
    "HOP_RUNG_EDGE",
    "HOP_RUNG_HUNT",
    "HOP_RUNG_LANE",
    "THREAT_RUNG_DUCK",
    "THREAT_RUNG_ROLLOUT",
    "THREAT_RUNG_REACTIVE",
    "PathNavPhase",
    "OverworldPathController",
    "is_5c_maze_hop",
    "MAZE_WAYPOINT_TOL",
    "SCREEN_5C_MAZE",
]
