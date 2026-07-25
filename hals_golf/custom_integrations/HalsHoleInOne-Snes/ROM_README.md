# Hal's Hole in One Golf ROM

Place `HalsHoleInOneGolf.smc` under `hals_golf/roms/` (or the shared
`retro_rl/roms/` folder). `hals_golf.runtime.retro_setup` repairs the
`rom.sfc` symlink from known locations.

Expected SHA1 (USA ROM, 512-byte SMC header stripped):

```
45baf328efa1e573aef81b2a936207f8979206a4
```

If you only have a `.smc` with a header, strip the first 512 bytes before
linking it as `rom.sfc`.
