# Hitbox recon — SM-K4.10-GATE / rr-dbu.10 (Double Chamber blue gate)

Scratch recon only. Does **not** STATUS-promote, claim Wave, or close the bead.
Controller geometry in `routes/kpdr/wave/` is owned by the controller agent.

## Sources

| Source | What it gives |
|--------|----------------|
| sm-json `refs/sm-json-data/region/norfair/east/Double Chamber.json` | Obstacle **A** = Blue Gate; room `0x7ADAD` / logic id 104 |
| Room diagram `…/roomDiagrams/east_DoubleChamber_104.png` | Gate mid upper path; switch box drawn **atop** vertical bars |
| Human tape `tasks/speed_to_ice_moat_human.json` f4650–5200 | Seat, volleys, first past-bars x |
| `debug/wave_recon/` (gate_switch*, human_full, human_replay) | Visual switch top; closed bars hard-stop; open vs closed frames |
| GHZ pattern `routes/kpdr/ghz_to_red.py` | Tight **Samus y** band + settle pose before shot |
| `scripts/probe/red_diag.py` | PLM open-state WRAM **blocked** (no trusted PLM record offset) |
| Pure gate open (promoted; residual purged) | Entry / Kamer / fail pin → controller `wave/` |

## Geometry truth (verified)

### Room / obstacle

- Room: Double Chamber **`0xADAD`** (sm-json address `0x7ADAD`).
- Obstacle **A** = **Blue Gate** (`obstacleType: inanimate`).
- Gate is a **vertical bar column** on the upper path; switch mechanism is the
  **gray box with blue LEDs at the TOP** of that column (not the bar faces).

### Solid bars vs switch

| Fact | Value | Evidence |
|------|------:|----------|
| Closed hard-stop (Samus center) | **x ≈ 411** upper path | human max x before open = 411; pure fail still stops here |
| Past-gate solid | **x ≳ 420–480, y ≈ 139** | human f5132 (421,139) pose 9; f5200 (477,139) |
| Switch side | **Top of bars** (ceiling unit) | diagram label A; all `wave_recon` screenshots; sm-json “shoot straight up” / “from below” |
| Expected projectile direction (upper seat) | **R-angle = diagonal up-right** (hold **R**, not UP+RIGHT) | human tape; horizontal RIGHT hits **bars** |
| Secondary direction | **Straight UP** from precisely under the gate | sm-json: “Shoot straight up to open the gate on the way up”; “from below without items” |

### Candidate switch hitbox (room pixels, approximate)

Not PLM-WRAM-confirmed. Image + hard-stop triangulation:

| Region | Room x | Room y | Notes |
|--------|-------:|-------:|-------|
| Bar column (solid when closed) | **400–416** | ~100–160 | Samus center stops ~411; bar face slightly right of center |
| Switch housing (visual top unit) | **~398–416** | **~80–100** | Top of bars; D0/gate_switch recon frames |
| Useful projectile impact band | **x ≈ 400–416** | **y ≈ 80–100** | Must hit **switch**, not mid-bar |

GHZ analogy: GHZ gates the **Samus y** band that produces the correct projectile
spawn line (886–889), not the switch’s absolute y. For DC upper seat the
matching band is:

| Parameter | GHZ (pattern) | DC (human open) |
|-----------|---------------|-----------------|
| Seat | pillar settle | Kamer top **x ∈ [368, 378]**, **y ≤ 145** (cycle 139↔219) |
| Pose before fire | stand settle (avoid landing 0xA4) | stand **pose 1/5**, not spin/landing |
| Fire y band | **886–889** rise | peak **Samus y ∈ [104, 111]** pose **105** |
| Aim | RIGHT + X | **R + X** (angle-up diagonal) |
| Open proof | walk x past pillar/gate | walk **x ≥ 413** then solid **x ≳ 420 y≈139** |

### Human open timeline (assist ammo — missiles never drain)

| frames | xy / sel | inputs | role |
|-------:|----------|--------|------|
| 4650 | (378,139) sel=1 | seat | Kamer high-ish |
| 4679–4709 | y 146–161 sel=1 | standing **X+R** | missiles; Kamer dropping |
| 4722–4731 | peak **y 108–111** p105 | **X+R** (+A) | missiles peak |
| 4834–4848 | y 122–160 p19 | pure **X** | fall-volley |
| 4964 / 5005 | sel **1→2→0** | SELECT | beam for final |
| **5035–5054** | peak **y 104–111** p105 sel=0 | **A+X+R / X+R** | **beam money volley** |
| 5083–5125 | approach B+RIGHT | still x≤411 | fuse / walk-up |
| **5126** | **(413,135)** p25 | B+RIGHT | first past bars (air) |
| **5132** | **(421,139)** p9 | B+RIGHT | solid walk — gate clear |
| 5206 | (494,139) | — | missile pack (+5); still ADAD |

Open is proven **after the beam peak volley**, not by bar sparks alone.
Human replay from pure pin under `debug/wave_recon/human_replay/` still shows
closed bars at x≈411 in several frames — seat/Kamer phase alignment matters.

## Bottom path bypass?

| Path | Opens A? | Pure KPDR useful for Wave door? |
|------|----------|----------------------------------|
| Upper Kamer seat + shoot switch | yes | **yes** (intended) |
| sm-json “from below / shoot straight up” | yes | possible alternate open; not required if upper works |
| G-mode despawn / CF under gate | despawn / glitch | **no** (not pure KPDR) |
| Bottom spikes → right climb without A | G-mode only in sm-json for node 3 | **no** normal pure bypass |

**Conclusion:** no clean pure bottom bypass to Wave Super door without opening
or despawning the gate. Stay on upper open.

## PLM WRAM

`red_diag.build_door_plm_snapshot` explicitly marks **plmRecords.status =
blocked** — no source-confirmed live WRAM for gate open bit. Open proof for
this card remains:

1. Walk **x ≳ 480** (or solid **x ≳ 420**) on upper path after shots, or
2. Frame dump showing bars retract / Samus past x411 without bounce.

Ammo drain with **assist OFF** is a useful *shot fired* signal only; human
tape used assist so missiles stayed 15.

## Probe scratch

One-off `dc_gate_plm_recon.py` was deleted after the gate recon. Product
controller is `routes/kpdr/wave/`; do not resurrect the scratch probe.

## Recommended single experiment (controller agent)

**One knob only** (do not stack angle + SELECT + seat at once):

> Tighten peak fire to a **GHZ-style Samus-y band**: fire **beam (sel=0)
> X+R only while `104 ≤ samus_y ≤ 111`** (pose 105 preferred) from Kamer seat
> **x ∈ [368, 378]** after stand settle; remove “fire as soon as y≤120”
> (current scaffold condition is effectively always-true once y≤120).

Why this knob:

1. Human open aligns with **beam** peak at **y 104–111**, not standing mid-bar missiles.
2. GHZ already proved blue-gate switches need a **few-pixel Samus-y line**.
3. Impacts on bars ≠ switch; wider peak window shoots the bar face or overshoots.
4. Alternate one-knob if preferred: exact human button RLE from a **Kamer-top pure mid-save** (f4650–5132 slice) — still one experiment, not more angle thrash.

### Acceptance for that experiment

- Pure from `post_single_to_double_chamber_pure.state`
- Gate open: upper solid **x ≳ 480** (or dual frame proof past bars)
- Does **not** claim Wave beam bit / Super door (rr-re9)

## Non-claims

- No STATUS promote / continuous re-record
- No Wave beam / rr-re9 close
- No Ice hop invent
- PLM open bit still unmapped
- Did not edit `wave/`
