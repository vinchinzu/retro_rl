# Room-enemy combat is an Overlay Skill, not a BossSegment

Bosses own the room (`BossStrategy` / `BossSegment`). Room enemies overlay a
movement hop through exactly two public functions in `combat/enemies`:
`list_enemies` scans the complete room table once, then `choose` applies
Engage / Avoid / Absorb / Ignore using hop-owned Intent. Putting Atomics (or
any corridor enemy) on `BossSegment` would steal geometry the hop already
owns. A third `routes/skills` primitive would bury Species and Contact next
to charge-shot. Unknown RAM ids remain represented but Ignore, so adding a
species or hop changes the implementation or Intent rather than widening the
interface. The old Wrecked Ship ice modules remain compatibility shims.
