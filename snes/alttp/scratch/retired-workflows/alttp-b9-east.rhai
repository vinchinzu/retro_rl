let meta = #{
    name: "alttp-b9-east",
    description: "See if the layer-1 floor at row 45 columns 33-38 connects to the Eastern landing.",
    when_to_use: "After alttp-b9-north. The 7px step to (4808, 5998) stayed on column 25 floor. Do not repeat it. Row 45 columns 33-38 are floor.",
    phases: [
        #{ title: "Count", detail: "two offline paths to the east floor, stairs masked" },
        #{ title: "Act", detail: "walk that pocket, or one step onto the agreed cut byte" },
        #{ title: "Record", detail: "correct column 33 and quote both counts" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["pocket_steps", "door_steps", "illegal_on_path", "goes_south", "cut_col", "cut_row", "cut_byte", "note"],
    "properties": #{
        "pocket_steps": #{ "type": "integer" },
        "door_steps": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "goes_south": #{ "type": "integer" },
        "cut_col": #{ "type": "integer" },
        "cut_row": #{ "type": "integer" },
        "cut_byte": #{ "type": "integer" },
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
    p += "Copy layer 1. Replace every stairs-kind byte with a legend wall byte, in the copy only. Do not edit nav.py, tilemap.py, or the fixture. Do not return the paths in b9_path.json, b9_path_check.json, b9_north.json, or b9_north_check.json.\n";
    p += "Origin is room_origin of world (4808, 6009). Poses are world pixels on the 8px phase. Start is a legal pose on the masked grid within 8px of (4808, 6009). The pocket is any legal pose whose LINK_BODY covers row 45 and one of columns 33 through 38. pocket_steps is the find_path length to the closest such pose. door_steps is the find_path length from that pose to a legal pose whose body covers door tile (31, 6). Use 0 when there is no pose.\n";
    p += "If any pose on a path has y greater than the path's start y, goes_south is 1 and that length becomes 0. Otherwise goes_south is 0. illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. If it is not 0, both lengths become 0.\n";
    p += "When pocket_steps is 0, set cut_col, cut_row, and cut_byte only if exactly one attribute byte touches both the landing component and the row-45 columns 33-38 component, and tile_kind of that byte is neither wall nor ledge. Unknown qualifies. 0x04 does not. Otherwise all three are -1. Column 25's 7px shuffle to y=5998 is already graded and is not a cut.\n";
    p += "The note must list row 45 columns 31 through 38 as byte and kind. Both prior notes say 31 and 32 are 0x27 unknown and 33 through 38 are 0x00 floor. If your grid disagrees, quote the bytes you read.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it.\n";
    p += "Write only ";
    p += out_name;
    p += ". Keys: pocket_steps, door_steps, illegal_on_path, goes_south, cut_col, cut_row, cut_byte, path (the pocket path, or the door path when door_steps is greater than 0; empty when pocket_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note.\n";
    return p;
}

fn act_prompt(mode) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". One emulator. Read pin_probe.open_pin and describe, and ram.read_snapshot, before you step. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_7. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Open Play_eog9_7. Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop.\n";
    p += "Describe Play_eog9_8 before you save anything. Do not overwrite Play_eog9_8.state or Play_eog9_1 through Play_eog9_7. A new pin may be Play_eog9_9 only when that file is absent.\n";
    p += "Stair claim, written before UP: north on column 31 to row 54, west to column 25, north to (4808, 6056), layer 0, room 0xB9. UP sets submodule 0x10. Landing hit: room 0xB9, layer 1, x within 8 of 4808, y within 8 of 6009, HP 28/32, sword 1, lamp 1, keys 0, follower 0. y=6005 and y=6009 are hits. Outside that band, stop and do not save.\n";
    if mode == "probe" {
        p += "Read cut_col, cut_row, and cut_byte from snes/alttp/docs/routes/full_clear/b9_east.json. Take one step of at most 8px from the landing toward that cell. Write the predicted x, y, layer, and room first. If tile_kind of that byte is wall or ledge, do not step. Do not step onto column 25's already graded pose (4808, 5998) unless that pose is the cut cell. Grade after the one step. Do not take a second step. Do not save the probe pose.\n";
    } else {
        p += "Read the path in snes/alttp/docs/routes/full_clear/b9_east.json. If the first point is more than 8px from the live landing, stop. Do not walk south to meet it. Follow the list. Halt when a pose is more than 8px off the next point, when y becomes greater than the landing y, or when room or layer changes before the door. ";
        if mode == "door" {
            p += "Entering room 0xA9 is success. Record the glance and do not explore 0xA9. Save Play_eog9_9 at the 0xA9 landing only when that file is absent.\n";
        } else {
            p += "Stop on the last path pose. Do not continue toward the door. Save Play_eog9_9 there only when the glance matches that pose within 8px, room stays 0xB9, layer stays 1, and the file is absent.\n";
        }
    }
    p += "Return walked (true only on a matched landing and a finished path or a 0xA9 entry), saved (true only if Play_eog9_9.state was created), room, layer, and note (xy, hp, sword, lamp, keys, follower, what Play_eog9_8 described as, prediction, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_east.json, b9_east_check.json, b9_north.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Column 33 is not byte 0x27. Both b9_north notes say row 45 columns 31 and 32 are 0x27 and columns 33 through 38 are 0x00 floor. Replace the sentence that calls columns 32-33 byte 0x27. The pose (4808, 5998) is a 7px shuffle on column 25 floor. It did not test row 45. The 105-step line is not body-legal. The 109-step lists go south and are not the route.\n";
    p += "Quote pocket_steps and door_steps from both east files. Quote the act note. If the files disagree, do not pick a path. If mode is door and the note entered 0xA9, the frontier is room 0xA9. If mode is blocked, say the east floor has no agreed approach.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance and the stair landing. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_east.json, and b9_east_check.json under snes/alttp.\n";
    p += "Count a problem if any file still says columns 32-33 are byte 0x27, calls (4808, 5998) a test of row 45, calls 105 steps body-legal, offers a 109-step list as the walk, sets approachXy, adds a north_ point, marks an ORDER row after 4 true, claims a credits tape or finished autoplay, tells anyone to press A, or marks AP-105 done. Also count a problem if residual.md does not quote pocket_steps from both east files. Quote each line.\n";
    p += "ok is true only when all six files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Column 33 is floor in the north notes, not 0x27. (4808, 5998) did not test row 45. The 105-step and 109-step lists are not the route. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote pocket_steps from both east files. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_east.json"),
        label: "b9-east",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_east_check.json"),
        label: "b9-east-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_pocket = -1;
let s_pocket = -1;
let g_door = -1;
let s_door = -1;
let g_illegal = -1;
let s_illegal = -1;
let g_south = -1;
let s_south = -1;
let g_byte = -1;
let s_byte = -1;
let g_col = -1;
let s_col = -1;
let g_row = -1;
let s_row = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.pocket_steps != () { g_pocket = counts[0].output.pocket_steps; }
    if counts[0].output.door_steps != () { g_door = counts[0].output.door_steps; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.goes_south != () { g_south = counts[0].output.goes_south; }
    if counts[0].output.cut_byte != () { g_byte = counts[0].output.cut_byte; }
    if counts[0].output.cut_col != () { g_col = counts[0].output.cut_col; }
    if counts[0].output.cut_row != () { g_row = counts[0].output.cut_row; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.pocket_steps != () { s_pocket = counts[1].output.pocket_steps; }
    if counts[1].output.door_steps != () { s_door = counts[1].output.door_steps; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.goes_south != () { s_south = counts[1].output.goes_south; }
    if counts[1].output.cut_byte != () { s_byte = counts[1].output.cut_byte; }
    if counts[1].output.cut_col != () { s_col = counts[1].output.cut_col; }
    if counts[1].output.cut_row != () { s_row = counts[1].output.cut_row; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_pocket >= 0 && g_pocket == s_pocket && g_door == s_door && g_illegal == 0 && s_illegal == 0 && g_south == 0 && s_south == 0 {
    if g_pocket > 0 && g_door > 0 {
        mode = "door";
    } else if g_pocket > 0 {
        mode = "pocket";
    } else if g_byte > 0 && g_byte == s_byte && g_col == s_col && g_row == s_row && g_col >= 0 && g_row >= 0 {
        mode = "probe";
    } else {
        mode = "blocked";
    }
} else if g_pocket >= 0 && s_pocket >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "door" || mode == "pocket" || mode == "probe" {
    let acted = agent(act_prompt(mode), #{
        label: "b9-east-act",
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
        label: "b9-east-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-east-skeptic",
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
            label: "b9-east-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    pocket_steps: g_pocket,
    door_steps: g_door,
    note: record_note,
});
