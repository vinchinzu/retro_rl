# Easiest SNES Games for AI Agents to Complete Through Emulator Control and RAM Inspection

The goal is not merely to train an agent that becomes competent after thousands
of attempts. The more interesting challenge is to build an agent that can
complete an entire SNES game in one uninterrupted evaluation run using:

- Emulator-controlled button inputs
- Frame stepping
- RAM inspection
- Save-state analysis during development
- Scripted policies, behavior trees, planning, or limited model inference
- Optional screen interpretation where RAM alone is insufficient

This creates something closer to a robotic software-engineering benchmark than
a conventional reinforcement-learning benchmark. The agent must first understand
the game, identify useful state variables, construct a strategy, and then
execute that strategy reliably from the title screen to the ending.

The canonical execution order and current evidence are tracked in
[STATUS.md](STATUS.md). That table distinguishes checkpoint-based segment
progress, continuous runs, and available videos.

## Define “One-Shot” Carefully

“One-shot” can mean several different things. A useful benchmark should
distinguish among them.

### Scripted One-Shot

Humans or an AI development system may inspect the game, map RAM, test
individual stages, and build scripts. The final evaluation consists of one
continuous run without save-state recovery.

This is the easiest and most practical starting point.

### Autonomous One-Shot

The agent receives the ROM, emulator API, screenshots, and RAM access, but no
handcrafted memory map or game-specific script. It must discover the controls
and game state, formulate a strategy, and complete the game.

This is dramatically harder.

### Generalized One-Shot

The agent must complete a previously unseen game using tools and strategies
learned from other SNES games.

This is the moonshot benchmark: game-playing agent meets reverse engineer meets
QA automation system.

For initial experiments, the **scripted** path is the right target.

**Practical rule for this repo:** do not treat “one uninterrupted evaluation
run” as a hard gate. Prefer:

1. Develop with save states, segment scripts, and retries.
2. Get a reliable stage / scene / track clear first.
3. Chain segments into longer runs when the pieces work.
4. Only later harden toward continuous title-to-credits evaluation.

Reliable automation beats one-shot purity early on.

Once a game advances beyond isolated segments, follow the shared
[scripted full-run development process](FULL_RUN_PROCESS.md). In particular,
“clears from a clean checkpoint” and “clears from the entry produced by the
continuous route” are separate milestones.

## What Makes a Good First Game?

The easiest games are not necessarily the games with the shortest human
completion times. A short game can still be difficult to automate if its state
is hidden, highly random, visually ambiguous, or dependent on precise timing.

The strongest candidates have most of the following properties:

- Linear progression
- Few meaningful actions
- Stable camera behavior
- Clear level-complete flags
- Low randomness
- No inventory management
- No large overworld
- Few branching decisions
- Simple boss patterns
- Generous health or continues
- Reliable player and enemy coordinates in RAM
- Limited reliance on text, dialogue, or abstract puzzles
- Fast recovery after mistakes

A fifteen-minute game with vision-heavy object searching may be harder to
generalize than a one-hour beat-’em-up whose correct policy is essentially
“walk right, align vertically, and attack.”

## Recommended Development Tiers

### Tier 0: Emulator and Pipeline Validation

These games are useful for proving that the agent can start the game, read
state, inject controls, detect progress, and reach an ending. They are not
necessarily impressive gameplay demonstrations.

#### The Great Waldo Search

Probably the easiest full-game proof of concept.

The game consists largely of static search screens. An agent can identify the
current scene, move the cursor to known coordinates, select the required
character, and advance.

Useful state:

- Current scene
- Cursor X/Y position
- Selected object
- Correct-answer flag
- Scene-completion flag

Advantages: very short, nearly deterministic, minimal action space, easy to
debug, little danger of losing.

Limitation: a completed bot may amount to a coordinate lookup table rather than
a general game-playing system. It validates the plumbing, not the intelligence.

#### Children’s Educational Titles

Games in the Mario’s Early Years series and similar educational software may
have simple menus, deterministic activities, and forgiving progression.

These can test menu navigation, cursor control, state-machine construction,
detecting success screens, and handling multiple small activities.

Their weakness is that many do not have a satisfying traditional “game
completed” endpoint.

### Tier 1: First Real Gameplay Benchmarks

These games provide movement, enemies, health, stages, and bosses without
exploding the state space.

#### Final Fight

One of the best first serious targets.

The game is linear, enemy behavior is readable, and progression generally
consists of moving right until enemies appear, defeating them, and continuing.

A workable controller could use a behavior tree:

- If enemies are present, align vertically.
- Move toward the closest enemy.
- Attack until the enemy’s health or presence flag disappears.
- Avoid being surrounded by repositioning.
- When no enemies remain, walk toward the right side of the screen.
- Trigger a boss-specific routine when the stage boss appears.

Useful state: player X/Y, player health, enemy slots and coordinates, enemy
health, enemy animation state, camera position, stage number, screen-lock flag,
boss-active flag.

Largest challenge: preventing the agent from standing slightly above or below
enemies and punching empty air.

#### TMNT IV: Turtles in Time

Similar advantages to Final Fight, but with faster movement and more varied
stage mechanics. Short, visually readable, and forgiving enough that a robust
but inelegant strategy can work.

#### Super Double Dragon

Short, linear, and built around a small combat vocabulary. Blocking adds
complexity, but an early agent could rely on aggressive attacks, health
monitoring, and conservative movement.

#### Rival Turf! or Brawl Brothers

Neither is especially glamorous, but both are structurally suitable: linear
stages, repeated combat loops, limited exploration, straightforward bosses.
Useful as second-game generalization tests after Final Fight.

### Tier 2: Deterministic Control Tasks

#### F-Zero

Possibly the best early racing benchmark. Racing games offer unusually useful
RAM variables: track position, lateral offset, speed, heading, lap, rank,
energy, collision state.

Development sequence: complete one track without crashing → record centerline
trajectory → optimize steering/boost → add collision recovery → generalize
across tracks → complete a league/cup.

#### Pilotwings

Excellent for mission-specific scripting. Each event has a limited objective
and relatively deterministic physics. Not one unified control problem
(skydiving, light-plane, rocketbelt, etc.), so good as a multi-skill benchmark
after the pipeline is stable.

#### Battle Clash

A Super Scope game can become surprisingly tractable when the emulator exposes
cursor coordinates and trigger inputs. Peripheral emulation adds engineering
complexity, but gameplay itself can be simpler than platforming.

### Tier 3: Forgiving Platformers

#### The Magical Quest Starring Mickey Mouse

One of the strongest early platforming candidates: relatively short, colorful,
forgiving, manageable transformations, mostly linear levels.

#### Disney’s Aladdin

Short and largely linear. Difficulties: momentum jumps, rope/platform
interactions, occasional fast autoscrolling, boss-specific behavior.

#### Tiny Toon Adventures: Buster Busts Loose!

Short and segmented into distinct stages. Useful for testing policy switching
by level identity.

#### Joe & Mac

Straightforward prehistoric action-platformer with ranged attacks and bosses.
More projectile chaos than Magical Quest, less navigational complexity than SMW.

#### Run Saber

Short, linear, mechanically simple by action-platformer standards. Plausible
compact full-game script candidate.

## Famous Games That Are Good Benchmarks but Poor First Targets

- **Super Mario World** — familiar but momentum, branching exits, overworld,
  power-ups, autoscrollers, bosses, and death recovery make full-game automation
  hard. Ideal framework benchmark after simpler platformers.
- **Super Castlevania IV** — linear but knockback, stairs, pits, whip
  direction, projectiles make it an intermediate target.
- **Star Fox** — Super FX / 3D instrumentation complexity; showcase later.
- **Wild Guns** — dense high-frequency aiming/dodging; advanced scripting.
- **Contra III** — short but brutally unforgiving; poor early target.

## Expanded Candidate Ranking

| Game | Agent Difficulty | Main Control Style | Determinism | Instrumentation | Recommended Role |
|------|------------------|--------------------|-------------|------------------|------------------|
| The Great Waldo Search | Very low | Cursor selection | Very high | Low | Pipeline proof |
| Final Fight | Low | Movement and melee | High | Low | First real completion |
| TMNT IV | Low–medium | Movement and melee | High | Low | Strong showcase |
| Super Double Dragon | Low–medium | Melee and blocking | High | Low | Combat generalization |
| F-Zero | Medium | Continuous steering | High | Low–medium | Control benchmark |
| The Magical Quest | Medium | Platforming | High | Medium | First platformer |
| Pilotwings | Medium | Continuous control | High | Medium | Multi-policy benchmark |
| Joe & Mac | Medium | Platforming + ranged | Medium–high | Medium | Action platformer |
| Aladdin | Medium | Momentum platforming | Medium–high | Medium | Platformer benchmark |
| Tiny Toon Adventures | Medium | Mixed stage mechanics | Medium | Medium | Policy switching |
| Super Mario World | Medium–high | Platforming + nav | High | Medium | Framework benchmark |
| Super Castlevania IV | High | Precision action | High | Medium | Advanced platformer |
| Star Fox | High | 3D rail shooting | Medium–high | High | Technical showcase |
| Wild Guns | High | Aiming and dodging | High | Medium–high | High-frequency control |
| Contra III | Very high | Precision run-and-gun | High | Medium | Stress test |

## Recommended First Five-Game Ladder

1. **The Great Waldo Search** — validate emulator startup, input injection,
   scene detection, progress flags, ending detection.
2. **Final Fight** — first real behavior tree: movement, enemies, health,
   stages, bosses.
3. **F-Zero** — continuous control, trajectory following, recovery.
4. **The Magical Quest** — reusable platforming primitives and room nav.
5. **Super Mario World** — scale platforming framework to overworld + longer
   sequences.

## Suggested Agent Architecture

### 1. Emulator Interface

Advance frames, press/release buttons, read RAM, capture screenshots, load
development save states, reset, record input movies. Prefer frame stepping
over real-time control.

### 2. State Extractor

Normalize raw memory into a common `GameState`:

```text
GameState
  game_mode, stage, room
  player_x, player_y, velocity_x, velocity_y
  health, lives
  camera_x, camera_y
  enemies[], projectiles[]
  boss_active, level_complete, player_dead
```

Each game has a small adapter into this structure so generic behaviors reuse:
move toward target, align with enemy, avoid projectile, jump over gap, wait
until grounded, detect death/room transition.

### 3. Controller Primitives

Higher-level actions instead of per-frame buttons:

- `walk_right(duration)` / `walk_left(duration)`
- `jump_right(duration)`
- `attack_until_clear()`
- `align_vertical(target_y)`
- `wait_until_grounded()`
- `move_to(x, y)`
- `dodge_projectile(projectile)`

### 4. Behavior Tree or Finite-State Machine

For early games, a behavior tree often beats a large neural policy because game
structure is explicit. Example:

```text
if player_dead: handle_continue()
elif level_complete: advance_menu()
elif boss_active: execute_boss_policy()
elif enemies_present: fight_nearest_enemy()
elif platforming_hazard_present: execute_navigation_segment()
else: move_toward_stage_exit()
```

### 5. Watchdog and Recovery System

Detect stuck states (no position change, repeating animation, enemy not taking
damage, walking into wall, stuck menu, camera stalled, health declining without
progress) and switch strategies / reposition / retry / deliberate death if
recoverable.

## RAM Discovery Workflow

1. **Differential search** — snapshot, move, compare changing addresses.
2. **Controlled perturbation** — change one property (health, room, jump,
   kill enemy) and diff before/after.
3. **Value freezing** — freeze candidates and observe effects.
4. **Correlation logging** — log candidates vs observed state over many frames.

## Better Benchmark Metrics

Track more than completion: evaluation attempts, deaths, continues, time,
frames, unique RAM addresses read, screenshot usage, whether game-specific
labels were provided, game-specific code volume, development save states,
human engineering time, AI-generated code percentage, robustness across resets,
timing perturbations, difficulty, and regional ROMs.

## Suggested Benchmark Classes

| Class | Description |
|-------|-------------|
| A | Input replay (fixed sequence baseline) |
| B | RAM-conditional script (best initial category) |
| C | RAM plus vision |
| D | Autonomous reverse engineering |
| E | Unseen-game generalization |

## Most Promising Initial Project

Cleanest serious proof of concept: **Final Fight** (Waldo proves plumbing;
Final Fight proves visible decisions). First milestone: development
`Stage1.state` (past character select) → clear one locked screen / fight
enemies reliably with a segment policy. Stretch later: chain segments into
full stage / stage boss, then harden toward continuous title-to-credits.

## SNES Agent Decathlon (long-term)

1. Cursor search — The Great Waldo Search
2. Beat-’em-up — Final Fight
3. Racing — F-Zero
4. Flight control — Pilotwings
5. Platforming — The Magical Quest
6. Overworld navigation — Super Mario World
7. Rail shooting — Star Fox
8. High-frequency aiming — Wild Guns
9. Precision action — Super Castlevania IV
10. Generalization — an unseen short licensed game

## Repo Mapping (this monorepo)

| Ladder slot | Game dir | Shared ROM zip |
|-------------|----------|----------------|
| 1 | `great_waldo_search/` | `roms/Super Nintendo/Great Waldo Search, The.zip` |
| 2 | `final_fight/` | `roms/Super Nintendo/Final Fight.zip` |
| 3 | `tmnt_iv/` | `roms/Super Nintendo/Teenage Mutant Ninja Turtles IV - Turtles in Time.zip` |
| 4 | `super_double_dragon/` | `roms/Super Nintendo/Super Double Dragon.zip` |
| 5 | `rival_turf/` | `roms/Super Nintendo/Rival Turf!.zip` |
| 6 | `f_zero/` | `roms/Super Nintendo/F-Zero.zip` |
| 7 | `magical_quest/` | `roms/Super Nintendo/Magical Quest starring Mickey Mouse, The.zip` |
| 8 | `pilotwings/` | `roms/Super Nintendo/Pilotwings.zip` |
| 9 | `battle_clash/` | `roms/Super Nintendo/Battle Clash.zip` |
| 10 | `joe_and_mac/` | `roms/Super Nintendo/Joe & Mac - Caveman Ninjas.zip` |

Shared oneshot helpers live in `snes_oneshot/`. Emulator plumbing stays in
`retro_harness/`. Elevate reusable primitives into `snes_oneshot/`; keep
game-specific RAM maps and policies inside each game directory.
