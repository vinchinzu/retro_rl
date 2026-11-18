"""Real-ROM acceptance: natural-entry opening, then exact controller replay."""

import json

import pytest

from snes.chrono_trigger.bot import OPENING, ROM, Runner, load_rows
from snes.chrono_trigger.maps import exit_graph, geometry_path

pytestmark = [pytest.mark.rom, pytest.mark.rom_smoke]


def test_power_on_clears_canyon_and_replays(tmp_path):
    if not ROM.is_file():
        pytest.skip("supply the US v1.0 ROM via chrono_trigger setup")
    # One uninterrupted predecessor chain; no checkpoint fixtures or restores.
    source = tmp_path / "opening"
    runner = Runner(output=source)
    assert runner.run([{**row, "pin": False} for row in OPENING])
    original = runner.state
    assert original.location == 0x1F1
    assert original.story == 12
    assert original.party == (0, 128, 128)
    assert 0xD6 in original.items
    assert original.exhibit_open and original.telepod_tested
    assert not original.candy_pending
    assert original.ready and original.control and not original.message
    assert 0 < original.hp < original.max_hp == 70
    assert original.mp == original.max_mp == 8
    assert original.gold == 509
    assert original.gate_battle_won
    assert original.last_battle_result == 0
    assert original.mode == "world" and not original.battle
    assert runner.state_loads == 0
    assert runner.frame < 40000
    # Dialogue confirmation must preserve the three default character names.
    assert runner.ram[0x2C23:0x2C35] == bytes.fromhex(
        "a2cbc8c7c800 acbacbc5be00 abcebcbcba00"
    )

    tape = load_rows(source / "inputs.json")
    replay = Runner(output=tmp_path / "replay")
    assert replay.run(tape)
    assert replay.frame == runner.frame
    assert replay.state == original
    assert replay.state_loads == 0
    assert replay.battle_events == runner.battle_events

    report = json.loads((source / "report.json").read_text())
    assert report["continuous_from_power_on"]
    assert report["mid_run_state_loads"] == report["ram_writes"] == 0
    assert not report["game_complete"]
    milestones = {r["id"]: r["snapshot"] for r in report["rows"]}
    assert milestones["MarleJoins"]["party"] == [0, 1, 128]
    assert not milestones["MarleJoins"]["exhibit_open"]
    assert milestones["ExhibitNews"]["exhibit_open"]
    assert milestones["CandyStop"]["candy_pending"]
    assert not milestones["CandyDone"]["candy_pending"]
    assert milestones["LuccaGreeting"]["story"] == 10
    assert milestones["LuccaGreeting"]["items"] == []
    assert not milestones["LuccaGreeting"]["telepod_tested"]
    assert milestones["TelepodLeft"]["telepod_tested"]
    assert milestones["MarleTelepod"]["story"] == 12
    assert milestones["MarleTelepod"]["location"] == 8
    assert milestones["MarleTelepod"]["items"] == []
    assert not milestones["FirstGate"]["gate_battle_won"]
    assert milestones["FirstAmbush"]["battle_phase"] == "command"
    assert milestones["FirstAmbush"]["hp"] == 70
    assert milestones["FirstBattle"]["gate_battle_won"]
    assert milestones["FirstBattle"]["gold"] == 436
    assert milestones["BridgeBattle"]["gold"] == 460
    assert milestones["LowerBattle"]["gold"] == 509
    assert len(milestones["BridgeAmbush"]["battlers"]) == 3
    assert len(milestones["LowerAmbush"]["battlers"]) == 4

    events = json.loads((source / "battles.json").read_text())
    assert {"starting", "command", "target", "resolving", "victory", "leaving"} <= {
        e["battle_phase"] for e in events
    }
    first_end = next(r["end_frame"] for r in report["rows"] if r["id"] == "FirstBattle")
    targets = {e["target_slot"] for e in events if e["battle_phase"] == "target" and e["frame"] < first_end}
    assert targets == {3, 4, 5}
    assert sum(e["battle_phase"] == "starting" for e in events) == 3
    battle_start = next(e for e in events if e["battle_phase"] == "command")
    assert [(b["slot"], b["identity"], b["hp"]) for b in battle_start["battlers"]] == [
        (0, 0, 70), (3, 12, 13), (4, 12, 13), (5, 12, 13),
    ]
    assert any(0 < e["hp"] < 70 and e["mode"] == "battle" for e in events)
    victory = next(e for e in events if e["battle_phase"] == "victory")
    assert all(b["hp"] == 0 for b in victory["battlers"] if b["slot"] >= 3)
    assert events[-1]["gate_battle_won"] and events[-1]["mode"] == "field"
    atlas = json.loads((source / "atlas.json").read_text())
    assert {2, 1, 0x1F0, 5, 0x1B7, 8, 0x71, 0x70, 0x1F1} <= {s["id"] for s in atlas["scenes"]}
    assert {(t["from"], t["to"]) for t in atlas["transitions"]} >= {
        (2, 1), (1, 0x1F0), (0x1F0, 5), (5, 0x1B7),
        (0x1B7, 5), (0x1B7, 8), (8, 0x71), (0x71, 0x70), (0x70, 0x1F1),
    }
    world = next(s for s in atlas["scenes"] if s["id"] == 0x1F0)
    assert any(t["destination"] == 5 and t["tile"] == [26, 14] for t in world["triggers"])

    graph = exit_graph(ROM.read_bytes())
    assert len(graph["locations"]) == 512
    assert sum(len(s["exits"]) for s in graph["locations"]) == 814
    assert geometry_path(graph, 2, 1) == [2, 1]
    assert graph["locations"][2]["exits"][0]["tile"] == [23, 13]


def test_first_battle_defeat_stops_attack_row(tmp_path):
    if not ROM.is_file():
        pytest.skip("supply the US v1.0 ROM via chrono_trigger setup")
    ambush = next(i for i, row in enumerate(OPENING) if row["id"] == "FirstAmbush")
    runner = Runner(output=tmp_path / "defeat")
    rows = [{**row, "pin": False} for row in OPENING[:ambush + 1]] + [
        {"id": "ObserveDefeat", "kind": "wait", "frames": 1, "pin": False,
         "goal": {"battle_phase": "defeat", "last_battle_result": 1}, "timeout": 40000},
        {"id": "RejectDefeat", "kind": "battle", "pin": False,
         "goal": {"ready": True, "control": True}, "timeout": 120},
    ]
    assert not runner.run(rows)
    assert runner.rows[-1]["reason"] == "party_defeated"
    assert runner.state.battle and runner.state.hp == 0
    assert runner.state.last_battle_result == 1
    # Persistent field HP remains stale while the battle record reaches zero.
    assert runner.ram[0x2603] == 70
    assert not runner.state.gate_battle_won


def test_field_control_and_zero_result_do_not_imply_victory(tmp_path):
    if not ROM.is_file():
        pytest.skip("supply the US v1.0 ROM via chrono_trigger setup")
    runner = Runner(output=tmp_path / "no_battle")
    assert not runner.run([
        {**OPENING[0], "pin": False},
        {"id": "RejectStaleResult", "kind": "battle", "pin": False,
         "goal": {"ready": True, "control": True}, "timeout": 48},
    ])
    assert runner.state.ready and runner.state.control
    assert runner.state.last_battle_result == 0
    assert runner.rows[-1]["reason"] == "timeout"


def test_power_on_route_clears_factory_and_replays(tmp_path):
    """Natural entries through Arris, Lab 32, Robo, the Factory and the End of Time."""
    if not ROM.is_file():
        pytest.skip("supply the US v1.0 ROM via chrono_trigger setup")
    from snes.chrono_trigger.route import ROUTE

    runner = Runner(output=tmp_path / "route")
    assert runner.run([{**row, "pin": False} for row in ROUTE])
    final = runner.state
    passed = {row["id"] for row in runner.rows if row["passed"]}
    assert {"Yakra", "Trial", "PrisonWait", "DragonTank", "Gate2300", "Guardian",
            "RatPassword", "LavosRecording", "BikeKey", "ArrisOut", "Lab32East",
            "RoboJoins", "ZabieY", "FactoryShutdown", "RSeries", "RoboRecovery"} <= passed
    assert runner.state_loads == 0
    assert final.story == 0x48 and final.location == 0x1D0
    assert final.party == (0, 1, 2)  # default selection; repaired Robo is in reserve
    assert final.ready and final.control and not final.message and not final.battle
    assert final.hp > 0
    assert 0xD7 in final.items  # the Gate Key from Lucca
    assert 0xD5 in final.items and 0xD4 not in final.items  # Seed exchanged for Bike Key
    assert runner.ram[0x100EC] & 0x40  # rat caught; password actually learned
    assert runner.ram[0x100A4] & 0xE0 == 0xE0  # all three L+R+A switches opened
    assert runner.ram[0x10154] & 8  # race attempt opened the foot crossing
    assert runner.ram[0x101D0] & 3 == 3  # Zabie password and shutdown escape
    milestones = {row["id"]: row["snapshot"] for row in runner.rows}
    assert milestones["Guardian"]["location"] == 0x158
    assert milestones["Guardian"]["story"] == 0x35  # unchanged by the win
    assert milestones["ArrisOut"]["story"] == 0x37
    assert milestones["RoboJoins"]["party"] == (0, 3, 2)
    assert milestones["ProtoRest"]["hp"] == milestones["ProtoRest"]["max_hp"]
    assert milestones["ProtoRest"]["mp"] == milestones["ProtoRest"]["max_mp"]
    assert milestones["FactoryShutdown"]["story"] == 0x3F
    assert milestones["RSeries"]["story"] == 0x42
    assert any(e["battle_phase"] == "victory" and any(
        b["identity"] == 0xD4 and b["slot"] >= 3 and b["hp"] == 0 for b in e["battlers"]
    ) for e in runner.battle_events)
    victories = [e for e in runner.battle_events if e["battle_phase"] == "victory"]
    assert any(len(robots := [b for b in e["battlers"] if b["slot"] >= 3 and b["identity"] == 0xF8]) == 6
               and all(b["hp"] == 0 for b in robots) for e in victories)
    # Healing/menu recovery can pause enemy damage longer than the escape timer.
    # Mandatory bosses must keep fighting through that pause.
    for boss in ("Yakra", "DragonTank", "Guardian", "RSeries"):
        index = next(i for i, row in enumerate(runner.rows) if row["id"] == boss)
        start, end = runner.rows[index - 1]["end_frame"], runner.rows[index]["end_frame"]
        assert not any(start <= p["frame"] < end and p["pick"][0] == "flee" for p in runner.picks)
    rseries_start = next(row["end_frame"] for row in runner.rows if row["id"] == "FactoryEscape")
    assert any(p["frame"] >= rseries_start and p["pick"][:2] == ["item", 0xC0] for p in runner.picks)
    # Steel Saber or better from the cathedral, put on by the outfit row.
    assert runner.ram[0x2600 + 0x29] >= 0x03
    assert runner.frame < 500000

    replay = Runner(output=tmp_path / "replay")
    assert replay.run(load_rows(tmp_path / "route" / "inputs.json"))
    assert replay.frame == runner.frame
    assert replay.state == final
    assert replay.battle_events == runner.battle_events
