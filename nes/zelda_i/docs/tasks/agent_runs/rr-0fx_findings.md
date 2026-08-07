# rr-0fx findings (runs 1–2)

## Verified

- Level3Complete settles → OW **0x74** x=128 y=125 **raft=1** tf=0x04
- Live hops that work: **0x74 LEFT → 0x73** (y≈141), **0x73 UP → 0x63** (arrives **y=221** south edge)
- On 0x63 after arrival: **UP 60f → y≈140** (into screen). RIGHT 60f @ y≈141 only moves to x≈119 — **does not cross** east (terrain / wrong band). Need longer push, different y-band, or alternate route (e.g. north then east).
- Wait for **mode 5** after every transition (mode 8 = scroll).
- Screenshots: recordings/l4rec_{0..4}_*.png

## DO NOT

- attach_image / vision thrash
- re-read LEVEL*_ROUTE docs
- poke Raft
- Clean STATUS

## DO NOW

1. Fix walk: after each screen change, idle until mode==5; after UP into 0x63, move **UP into room** to y≈141 (or DOWN from south edge if needed), then RIGHT to 0x64 @y141.
2. Continue hops to dock **0x55**, walk dock with Raft → island **0x45** (or live override), enter door, record entry room.
3. Implement `OverworldToLevel4Controller` via `ow_path.OverworldPathController` (copy L5/L3 patterns).
4. 2/2 runner + Level4Entrance.state + recordings/l4_entry_recon.json
5. Update LEVEL4_ROUTE.md live ids; bd close rr-0fx only if 2/2
6. Delete _tmp_* scripts

Use only ipython shell tools. RAM/coords only — no images.
