# TASK SM-SPAZER-POLICY: Prefer Spazer in later policies when held

## Recipe step
primitive promote / efficiency (combat policy)

## Model
Luna

## Wave type
implement

## Own files only
- Combat / spray helpers that choose beam mix (narrow list in residual after
  inventory of call sites)
- Tests locking “if Spazer collected → equip/use path”
- residual required (name touched files)

Depends on: `SM-SPAZER-PURE` green (tip optional).

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Early Spazer only pays off if later controllers **use** it.
- Do not break no-Spazer minimal spine until FOLD; policies must branch on
  collected beams.

## Read first
- Kraid / missile spray combat modules
- `ram.py` beam fields
- Mother Brain note: plasma conflicts with Spazer (late game)

## Do
1. Inventory call sites that hard-code missile-only or Charge-only spray.
2. Add one safe branch: when Spazer collected and not conflicting late beam,
   equip/use Spazer for wide spray (Kraid or a single shared helper first).
3. Unit tests with mock state beams.

## Do not
- Force equip Spazer when not collected
- Plasma+Spazer illegal combo late without explicit unequip rule
- Continuous tip rewrite

## Acceptance
- [ ] At least one later policy path uses Spazer when held
- [ ] No-Spazer path still works
- [ ] Tests green
