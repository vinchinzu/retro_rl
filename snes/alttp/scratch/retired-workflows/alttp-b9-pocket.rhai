let meta = #{
    name: "alttp-b9-pocket",
    description: "From Play_eog9_9, count the door and test one step north of the east floor.",
    when_to_use: "After alttp-b9-east. The grade held (4872, 5984). door_steps is 0 and the cut was never named. Do not replay the stair.",
    phases: [
        #{ title: "Count", detail: "two offline door counts from the pocket" },
        #{ title: "Act", detail: "open Play_eog9_9, walk the door or step north once" },
        #{ title: "Record", detail: "quote the pin and the north byte" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["door_steps", "illegal_on_path", "goes_south", "north_col", "north_row", "north_byte", "note"],
    "properties": #{
        "door_steps": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "goes_south": #{ "type": "integer" },
        "north_col": #{ "type": "integer" },
        "north_row": #{ "type": "integer" },
        "north_byte": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

let act_schema = #{
    "type": "object",
    "required": ["walked", "saved", "room", "layer", "note"],
    "properties": #{
        "walked": #{ "type": "boolean" },
        "saved": #{ "type": "boolean" },
        "room": #{ "type": "string" },
        "layer": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

let record_schema = #{
    "type": "object",
    "required": ["path", "note"],
    "properties": #{
        "path": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
};

let skeptic_schema = #{
    "type": "object",
    "required": ["ok", "problems", "note"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "problems": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

fn count_prompt(out_name) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Offline only. Do not boot a ROM. Do not run room_engine. Do not press A.\n";
    p += "Read find_path and room_origin in snes/alttp/dungeon/nav.py, tile_kind in snes/alttp/dungeon/tilemap.py, and LINK_BODY in snes/alttp/walk_physics.py. Read snes/alttp/tests/fixtures/room_b9_layer1.json. Grids are 64 by 64, index [row][col].\n";
    p += "Copy layer 1. Replace every stairs-kind byte with a legend wall byte, in the copy only. Do not edit nav.py, tilemap.py, or the fixture. Do not return the paths in b9_path.json, b9_path_check.json, b9_north.json, b9_north_check.json, b9_east.json, or b9_east_check.json.\n";
    p += "Origin is room_origin of world (4872, 5984). Poses are world pixels on the 8px phase. Start is a legal pose on the masked grid within 8px of (4872, 5984). door_steps is the find_path length to a legal pose whose body covers door tile (31, 6). Use 0 when there is no pose.\n";
    p += "If any pose has y greater than the start y, goes_south is 1 and door_steps is 0. Otherwise goes_south is 0. illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. If it is not 0, door_steps is 0.\n";
    p += "north_col, north_row, and north_byte are the masked-grid cell one body-north of the start, the cell the next 8px north step would put LINK_BODY on. The note must give that byte and its tile_kind, and the bytes of that whole row for columns 28 through 40.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it.\n";
    p += "Write only ";
    p += out_name;
    p += ". Keys: door_steps, illegal_on_path, goes_south, north_col, north_row, north_byte, path (empty when door_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note.\n";
    return p;
}

fn act_prompt(mode) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". One emulator. Open Play_eog9_9. Do not open Play_eog9_7. Do not replay the stair. Read pin_probe.open_pin and describe, and ram.read_snapshot, before you step. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_9. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop. Do not overwrite Play_eog9_1 through Play_eog9_9.\n";
    p += "Describe Play_eog9_9 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5984, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop. Do not walk back to the landing.\n";
    if mode == "probe" {
        p += "Read north_col, north_row, and north_byte from snes/alttp/docs/routes/full_clear/b9_pocket.json. Write the predicted x, y, layer, and room before the step. If tile_kind of north_byte is wall or ledge, predict that xy stays within 8px and still take the one step so the grade can fail. If the kind is floor or unknown, predict one step of at most 8px north. Take that one step. Grade it. Do not take a second step. Do not walk south. Save Play_eog9_10 only when the step moved, the grade matched, the file is absent, room stays 0xB9, and layer stays 1.\n";
    } else {
        p += "Read the path in snes/alttp/docs/routes/full_clear/b9_pocket.json. If the first point is more than 8px from the live pose, stop. Do not walk south to meet it. Follow the list. Halt when a pose is more than 8px off the next point, when y becomes greater than the pin y, or when room or layer changes before the door. Entering room 0xA9 is success. Record the glance and do not explore 0xA9. Save Play_eog9_10 at that landing only when the file is absent.\n";
    }
    p += "Return walked (true only for a finished door path or a 0xA9 entry), saved (true only if Play_eog9_10.state was created), room, layer, and note (the Play_eog9_9 glance, xy, hp, sword, lamp, keys, follower, prediction, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_pocket.json, b9_pocket_check.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file. Also read snes/alttp/custom_integrations/Zelda3-Snes/Play_eog9_9.provenance.json.\n";
    p += "Play_eog9_9.state exists. Provenance kind is resumed, source_run Play_eog9_7, frame 189. The grade that saved it held (4872, 5984), layer 1, room 0xB9. The current residual omitted that save. Put it back. The pocket is not a north-door approach. door_steps from the pocket was 0 in b9_east.json and b9_east_check.json because the cut was not named.\n";
    p += "Quote door_steps, north_col, north_row, and north_byte from both pocket files. Quote the act note. If the files disagree, do not pick a path. If mode is walk and the note entered 0xA9, the frontier is room 0xA9.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, and the pocket grade. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_pocket.json, and b9_pocket_check.json under snes/alttp.\n";
    p += "Count a problem if Play_eog9_9 is missing from residual.md, if door_steps from both pocket files are not quoted, if (4872, 5984) is called a door approach, if 105 steps is called body-legal, if a 109-step list is offered as the walk, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all six files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_9 exists, kind resumed. (4872, 5984) is not a door approach. The 105-step and 109-step lists are not the route. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote door_steps from both pocket files. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_pocket.json"),
        label: "b9-pocket",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_pocket_check.json"),
        label: "b9-pocket-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_door = -1;
let s_door = -1;
let g_illegal = -1;
let s_illegal = -1;
let g_south = -1;
let s_south = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.door_steps != () { g_door = counts[0].output.door_steps; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.goes_south != () { g_south = counts[0].output.goes_south; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.door_steps != () { s_door = counts[1].output.door_steps; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.goes_south != () { s_south = counts[1].output.goes_south; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_door >= 0 && g_door == s_door && g_illegal == 0 && s_illegal == 0 && g_south == 0 && s_south == 0 {
    if g_door > 0 {
        mode = "walk";
    } else {
        mode = "probe";
    }
} else if g_door >= 0 && s_door >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "walk" || mode == "probe" {
    let acted = agent(act_prompt(mode), #{
        label: "b9-pocket-act",
        capability_mode: "all",
        output_schema: act_schema,
    });
    if acted != () && acted.success && type_of(acted.output) == "map" && acted.output.note != () {
        act_note = acted.output.note;
    }
}
log(act_note);

phase("Record");
let record_note = "no record";
if mode != "unverified" {
    let recorded = agent(record_prompt(mode, act_note), #{
        label: "b9-pocket-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-pocket-skeptic",
        capability_mode: "read-only",
        output_schema: skeptic_schema,
    });
    let problems = 0;
    let skeptic_note = "";
    if skeptic != () && skeptic.success && type_of(skeptic.output) == "map" {
        if skeptic.output.problems != () { problems = skeptic.output.problems; }
        if skeptic.output.note != () { skeptic_note = skeptic.output.note; }
    }
    if problems > 0 {
        agent(fix_prompt(skeptic_note), #{
            label: "b9-pocket-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    door_steps: g_door,
    check_steps: s_door,
    note: record_note,
});
