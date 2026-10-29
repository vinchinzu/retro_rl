# Plan: Harvest Moon (SNES)

Future work. Verified results: [STATUS.md](STATUS.md). Campaign detail:
[MASTER_PLAN.md](perfect_rancher/MASTER_PLAN.md), tracker IDs in
[BEAD_PLAN.md](perfect_rancher/BEAD_PLAN.md). Queue:
`bd ready -l harvest -l rancher999`.

## Repair pass, 2026-09-30

Merged fixes now feed actual cow slots, retain unfinished milestone intents,
select native tasks before the advisor, and require immutable RAM receipts.
Use `run_to_day2 --perfect-rancher` to enable campaign dispatch; ordinary
farm runs retain their rule order. Unsupported acquisitions defer explicitly.
A cow receipt requires a newly initialized slot as well as the count change.

Merged native evidence:

- Twelve-cow feed/night fixture: 7,507 frames, feed flags 0x0FFF,
  hay 500→488, all twelve happiness values remain 200 after the Spring D16→17 sleep.
  The alleged missing slot-0 stall was a stale filled tile in an edited
  research pin. Native barn re-entry restores the trough. No replacement
  calf is required for slot 0; acquisition and fifty care days remain open.
- Dispatch fixture: legacy and milestone chicken purchases each take
  4,028 frames, count 0→1, wallet 4,090→3,090G, verified native receipt.
- Structure comparison: baseline, seeded changes and folded reporter each
  take 17,538 frames on the same D3 farm day, with identical end states.
  Runner/report/composer/multi-day sources meet the 1,000-line bar.

These are fixture proofs with zero live writes, mid-run loads or assists.
They do not establish natural campaign arrival or a 999 result.

## Next blockers

| Priority | Beads | Required evidence |
|----------|-------|-------------------|
| P0 | Exact-ten berries `rr-ux6x.16.7` | Execute native 7→10, guard 11 and preserve count/stamina after final sleep. The cabin quote/arm/clock gift now have a native first-house receipt below. |
| P0 | F7 `rr-ux6x.7.8`, F5 `rr-ux6x.7.6`, solver `rr-ux6x.8.2` | Validate one shared dated budget. Conditional YAML proposes marriage at index 128 plus 105 food repairs costing 31,500G; buy-three/breed-nine is an alternative. Repeated food, herd naming/feed and the complete Perfect cash schedule remain unproved. Basic first-house lumber and payment are received below. |
| P0 | Contract `rr-ux6x.4` | Resolve the handoff's parked questions versus ADR 0007's recorded duplication/load/fallback rulings. Preserve the recorded ADR; no new owner ruling is inferred. |
| P0 | Structure behavior `rr-ux6x.18.1` | Full ending requirement remains 2,175,350 frames. The 3,476,992 run changed policies/features, including daily clearing; the matched one-day proof passes. Do not close the full-run gate from that short comparison. |
| P1 | Dispatch `rr-ux6x.8.3`, arbiter `.8.4`, herd `.11.3/.11.5/.11.8`, social `.14.6/.14.7` | Current dispatch proof covers one purchase. Prove all due native domains and a real 12-cow/12-chicken day, measured care/medicine, and social windows. Model checks remain coverage only. |

Wife is Eve. Marriage can land in Fall or Winter. A cow comes first:
pasture stays on the calendar, and the first cow buys at 5,300G instead of
waiting behind the 6,000G house-and-feather fund. The carpenter payment,
including the 10,000G second house, waits until that cow exists. A Fall or
Winter first-house order does not arm the clock.

The first-house/marriage route now has native receipts
([EVE_BAR.md](EVE_BAR.md)): power-on festival decline preserves the wallet;
axe work banks 252 wood; corn funds the 5,000G/250-wood order on Summer D28;
four nights complete the house on Fall D2; the natural Sunday seller sells
the feather on Fall D3. The isolated wedding continuation finishes on Fall
D10 with Eve's wife bit set and retains it through two more nights. These
are Clean continuations from real power-on predecessors, with zero RAM writes
or mid-run loads. That route spent the summer wallet before any cow. The
Fall D18 resume (`LN_house2_wood_natural`) still has 0 cows, 680G, and
`wife_pregnancy` 9. The full campaign and Perfect result remain open.

The cabin's introduction needs a native exit/re-entry before the request
sign. The Summer order arms the clock, and the completion scene gives it;
`$7F1F6E & 0x4000` is set on the natural wedding checkpoint. First-house
materials, request, payment, construction, feather and wedding each have a
ROM check in `tests/test_house_upgrade.py`. House upgrade 2 is quoted
(500 wood, 10,000G) and one mountain chop from the married pin is in
[EVE_BAR.md](EVE_BAR.md); the order and the 450/650 pregnancy floors are
still open ([F4](perfect_rancher/research/F4_family.md)). Exact-ten berries and the
complete dated Perfect budget still need their own campaign receipts.

The [F10 verdict](perfect_rancher/research/F10_verdict.md) remains conditional:
seven designed berries give a table ceiling of 984 if every other term is
maxed. Neither 984 nor 999 has a verified campaign route. The calendar's
15 sightings cover five girls on three sampled days; native interactions
remain to be composed.

## Constraints

Wife hearts must be 498–511 before hour-18 entry subtracts 2; Maria's
25 poison-mushroom offers measure 660→510, but the real wife approach is
still needed. Wife bit by Y2 Fall D9 suffices for children present at D29;
index 128 is the conditional earlier target for child 2's age-60 bonus.
Development needs 2,950 of 2,968 cover candidates after the final night.
Girls and crop counts need 496–511, happiness ≥992, money
99,840–655,350G after overnight credit, twelve cows ≥200 and twelve chickens.
Missed Fall D20 prizes need recovery; berry count above ten fails the best
ending check. Advance year checkpoints only from their real predecessors.

## Existing farm spine

Keep short-window work in
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md): summer-entry
Gate B (`rr-5in`), weak spring yield (`rr-bu62.25`), cow talk timeout
(`rr-bu62.30`), crop stand weeds (`rr-bu62.8`), name-entry warps
(`rr-bu62.18`/`.26`), dog release (`rr-bu62.21`/`.23`), and the money
visitor (`rr-bu62.14`). Purchase success must include a new initialized
cow slot after naming, not only wallet/count changes. The seasonal purchase
recipe remains [livestock_purchase_plan.md](livestock_purchase_plan.md).

The earlier 2,175,350-frame working-farm credits proof remains in STATUS.
The audit changes planning and evidence claims; it does not promote a
Perfect result or replace the verified product baseline.
