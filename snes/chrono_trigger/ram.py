"""Read-only US v1.0 observations. Field DP is $0100, not $0000.

Sources and verification limits are recorded in docs/ram_map.md. Read the
actual WRAM block: get_ram() aliases high WRAM incorrectly on this core.
"""

from dataclasses import asdict, dataclass


def word(ram: bytes, offset: int) -> int:
    return int.from_bytes(ram[offset:offset + 2], "little")


@dataclass(frozen=True)
class Battler:
    slot: int
    identity: int
    present: bool
    hp: int
    max_hp: int
    mp: int
    max_mp: int


@dataclass(frozen=True)
class Snapshot:
    location: int
    x: float | None
    y: float | None
    control: bool
    ready: bool
    message: int
    naming: bool
    menu: bool
    battle: bool
    story: int
    party: tuple[int, ...]
    hp: int
    max_hp: int
    gold: int
    items: tuple[int, ...]
    exhibit_open: bool
    candy_pending: bool
    telepod_tested: bool
    mode: str
    mp: int
    max_mp: int
    battle_phase: str | None
    battlers: tuple[Battler, ...]
    command_actor: int | None
    command_row: int | None
    target_slot: int | None
    last_battle_result: int
    gate_battle_won: bool

    def record(self) -> dict:
        return asdict(self)


def read_wram(env) -> bytes:
    return bytes(env.data.memory.blocks[0x7E0000])


def observe(ram: bytes) -> Snapshot:
    if len(ram) != 0x20000:
        raise ValueError("expected the complete 128 KiB SNES WRAM block")
    location = word(ram, 0x100)
    menu = ram[0x500:0x504] == bytes.fromhex("5c0e84c2")
    naming = menu and ram[0x67] == 0x0D and 0x2C23 <= word(ram, 0xF0C) < 0x2C53
    nmi = ram[0x500:0x504]
    # The battle entry vector is in C1, but the installed NMI runs in D1.
    battle = nmi in (bytes.fromhex("5c7ce7d1"), bytes.fromhex("5cc0f4d1"))
    control = bool(ram[0x11F])
    overworld = location in range(0x1F0, 0x1F8)
    mode = ("battle" if battle else "menu" if menu else
            "world" if overworld and ram[0x503] == 0xC2 else
            "field" if not overworld and ram[0x503] == 0xC0 else "unknown")
    ready = not menu and not battle and (
        ram[0x503] == 0xC2 and ram[0x1BF6] & 3 == 0 and ram[0x1D] == 15 if overworld
        else ram[0x503] == 0xC0 and ram[0x119] == 15 and ram[0x117] == 0
    )
    x = y = None
    if mode in {"world", "field"}:
        if overworld:
            x, y = float(word(ram, 0x283)), float(word(ram, 0x285))
        elif location < 0x1F0:
            slot = word(ram, 0x197)
            if slot < 0x80 and slot % 2 == 0:
                x = word(ram, 0x1800 + slot) / 16
                y = word(ram, 0x1880 + slot) / 16
    battlers = ()
    phase = None
    actor = command_row = target = None
    hp, max_hp = word(ram, 0x2603), word(ram, 0x2605)
    mp, max_mp = word(ram, 0x2607), word(ram, 0x2609)
    if battle:
        # AF0A retains the initial roster when AEFF/present are cleared on KO.
        # Mode 0 is graphics setup; its stat buffers are not live yet.
        if ram[0x7E] in (1, 2):
            battlers = tuple(
                Battler(slot, ram[0xAF0A + slot], ram[0x96F5 + slot] == 1,
                        *(word(ram, 0x5E30 + slot * 0x80 + i * 2) for i in range(4)))
                for slot in range(11) if ram[0xAF0A + slot] != 0xFF
                and ram[0x96F5 + slot] in (0, 1)
            )
        pcs = [b for b in battlers if b.slot < 3]
        enemies = [b for b in battlers if b.slot >= 3]
        crono = next((b for b in pcs if b.identity == 0), None)
        if crono:
            hp, max_hp, mp, max_mp = crono.hp, crono.max_hp, crono.mp, crono.max_mp
        active = ram[0x95D5]
        if any(b.slot == active and b.present and b.hp > 0 for b in pcs) and ram[0x93EE + active * 7] & 0x80:
            actor = active
            command_row = ram[0x95DC + active]
        if actor is not None and ram[0x9609]:
            cursor = ram[0x9614]
            candidate = ram[0x99C0 + cursor] if cursor < 11 else 0xFF
            target = candidate if candidate < 11 else None
        phase = ("leaving" if nmi == bytes.fromhex("5cc0f4d1") else
                 "starting" if not pcs or not enemies else
                 "defeat" if all(b.hp == 0 for b in pcs) else
                 "victory" if all(b.hp == 0 for b in enemies) else
                 "target" if target is not None else
                 "command" if actor is not None and ram[0x95DB] == 0 else
                 "submenu" if actor is not None else "resolving")
    return Snapshot(
        location, x, y, control, ready, ram[0x129], naming, menu, battle,
        ram[0x10000], tuple(ram[0x2980:0x2983]),
        hp, max_hp,
        int.from_bytes(ram[0x2C53:0x2C56], "little"),
        tuple(ram[0x2400 + i] for i in range(242) if ram[0x2400 + i] and ram[0x2500 + i]),
        bool(ram[0x10055] & 0x80), bool(ram[0x10055] & 0x08),
        bool(ram[0x10056] & 0x02),
        mode, mp, max_mp, phase, battlers, actor, command_row, target,
        ram[0x101EC], bool(ram[0x10146] & 1),
    )
