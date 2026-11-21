let meta = #{
    name: "alttp-b9-span",
    description: "Flood the layer-1 floor north of Play_eog9_10 and walk that span once.",
    when_to_use: "After alttp-b9-pocket. The 7px step onto row 44 floor held. Do not repeat it. Name the north edge or the stair that touches it.",
    phases: [
        #{ title: "Count", detail: "two offline floods from the pocket" },
        #{ title: "Act", detail: "open Play_eog9_10 and walk the span or press UP once" },
        #{ title: "Record", detail: "quote the north edge and the pin" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["door_steps", "span_steps", "illegal_on_path", "goes_south", "edge_row", "stair_col", "stair_row", "stair_byte", "note"],
    "properties": #{
        "door_steps": #{ "type": "integer" },
        "span_steps": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "goes_south": #{ "type": "integer" },
        "edge_row": #{ "type": "integer" },
        "stair_col": #{ "type": "integer" },
        "stair_row": #{ "type": "integer" },
        "stair_byte": #{ "type": "integer" },
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
    p += "Copy layer 1. Replace every stairs-kind byte with a legend wall byte, in the copy only. Do not edit nav.py, tilemap.py, or the fixture. Do not return an older b9_path, b9_north, b9_east, or b9_pocket list.\n";
    p += "Origin is room_origin of world (4872, 5977). Poses are world pixels on the 8px phase. Start is a legal pose on the masked grid within 8px of (4872, 5977). Walkable kinds for the flood are floor, shallow_water, door, and ice.\n";
    p += "door_steps is the find_path length to a legal pose whose body covers door tile (31, 6). Use 0 when none exists. edge_row is the smallest row number in the start's walkable component. span_steps is the find_path length to the legal pose in that component with the smallest row, ties broken by closeness to column 33. If that pose is not at least 16px north of the start (its y is greater than start y minus 16), span_steps is 0. A 7px shuffle is not a span.\n";
    p += "If any pose has y greater than the start y, goes_south is 1 and both lengths become 0. Otherwise goes_south is 0. illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. If it is not 0, both lengths become 0.\n";
    p += "On the unmasked grid, stair_col, stair_row, and stair_byte name the stairs-kind cell that touches the component and has row less than or equal to edge_row. If several, pick the smallest row, then the column closest to 33. If none, all three are -1. Do not treat a stair as floor.\n";
    p += "The note must list every byte and tile_kind on the unmasked row edge_row minus 1, for the component's columns expanded by two on each side. Also say how many stairs-kind bytes were masked.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it.\n";
    p += "Write only ";
    p += out_name;
    p += ". Keys: door_steps, span_steps, illegal_on_path, goes_south, edge_row, stair_col, stair_row, stair_byte, path (the door path when door_steps is greater than 0, else the span path, else empty), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note.\n";
    return p;
}

fn act_prompt(mode) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". One emulator. Open Play_eog9_10. Do not open Play_eog9_7. Do not replay the stair. Do not repeat the 7px step from (4872, 5984) to (4872, 5977). Read pin_probe.open_pin and describe, and ram.read_snapshot, before you step. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_10. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop and report the sprite slots. If xy stops matching while HP stays 28, stop and report the sprite slots. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_10.\n";
    p += "Describe Play_eog9_10 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5977, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    if mode == "stair" {
        p += "Read stair_col, stair_row, and stair_byte from snes/alttp/docs/routes/full_clear/b9_span.json. Stand on the legal floor cell adjacent to that stair, no farther than 8px from the pin if the path is empty. Write the predicted room, layer, xy, and submodule before you hold UP. Hold UP once through submodule 0x10 or until control returns, at most 480 frames. Grade it. Do not take another step. Save Play_eog9_11 only when the grade shows a room or layer or xy change, the file is absent, and HP is still 28.\n";
    } else {
        p += "Read the path in snes/alttp/docs/routes/full_clear/b9_span.json. If the first point is more than 8px from the live pose, stop. Do not walk south to meet it. Follow the list. Halt when a pose is more than 8px off the next point, when y becomes greater than the pin y, or when room or layer changes before the door. ";
        if mode == "door" {
            p += "Entering room 0xA9 is success. Record the glance and do not explore 0xA9. Save Play_eog9_11 at that landing only when the file is absent.\n";
        } else {
            p += "Stop on the last path pose. If b9_span.json names a stair and the live pose is within 8px of the floor cell south of that stair, hold UP once, at most 480 frames, and grade submodule, room, layer, and xy. Otherwise do not press UP. Save Play_eog9_11 only when the file is absent, HP is 28, and the pose is at least 16px north of (4872, 5977) or the room changed.\n";
        }
    }
    p += "Cannonballs were described near (4888, 5732) and (4872, 5817). Report them if they are still there. Do not fight them.\n";
    p += "Return walked (true only for a finished path, a stair transition, or a 0xA9 entry), saved (true only if Play_eog9_11.state was created), room, layer, and note (the Play_eog9_10 glance, xy, hp, sword, lamp, keys, follower, prediction, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_span.json, b9_span_check.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file. Also read Play_eog9_10.provenance.json under snes/alttp/custom_integrations/Zelda3-Snes.\n";
    p += "Play_eog9_10.state exists. Provenance kind is resumed, source_run Play_eog9_9, frame 5. The grade was 5 UP frames from (4872, 5984) to (4872, 5977), layer 1, room 0xB9, HP 28. That step stood on floor. It is not a door approach. Put the pin in the residual. The 7px step must not be described as the route.\n";
    p += "Quote door_steps, span_steps, edge_row, and the stair cell from both span files. Quote the act note. If the files disagree, do not pick a path. If mode is door and the note entered 0xA9, the frontier is room 0xA9.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, and Play_eog9_10. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_span.json, and b9_span_check.json under snes/alttp.\n";
    p += "Count a problem if Play_eog9_10 is missing from residual.md, if span_steps from both span files are not quoted, if (4872, 5977) is called a door approach, if the 7px step is called the route, if 105 steps is called body-legal, if a 109-step list is offered as the walk, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all six files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_10 exists, kind resumed. (4872, 5977) is not a door approach. The 7px step is not the route. The 105-step and 109-step lists are not the route. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote span_steps from both span files. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_span.json"),
        label: "b9-span",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_span_check.json"),
        label: "b9-span-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_door = -1;
let s_door = -1;
let g_span = -1;
let s_span = -1;
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
    if counts[0].output.door_steps != () { g_door = counts[0].output.door_steps; }
    if counts[0].output.span_steps != () { g_span = counts[0].output.span_steps; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.goes_south != () { g_south = counts[0].output.goes_south; }
    if counts[0].output.stair_byte != () { g_byte = counts[0].output.stair_byte; }
    if counts[0].output.stair_col != () { g_col = counts[0].output.stair_col; }
    if counts[0].output.stair_row != () { g_row = counts[0].output.stair_row; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.door_steps != () { s_door = counts[1].output.door_steps; }
    if counts[1].output.span_steps != () { s_span = counts[1].output.span_steps; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.goes_south != () { s_south = counts[1].output.goes_south; }
    if counts[1].output.stair_byte != () { s_byte = counts[1].output.stair_byte; }
    if counts[1].output.stair_col != () { s_col = counts[1].output.stair_col; }
    if counts[1].output.stair_row != () { s_row = counts[1].output.stair_row; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_door >= 0 && g_door == s_door && g_span == s_span && g_illegal == 0 && s_illegal == 0 && g_south == 0 && s_south == 0 {
    if g_door > 0 {
        mode = "door";
    } else if g_span > 0 {
        mode = "span";
    } else if g_byte > 0 && g_byte == s_byte && g_col == s_col && g_row == s_row && g_col >= 0 && g_row >= 0 {
        mode = "stair";
    } else {
        mode = "blocked";
    }
} else if g_door >= 0 && s_door >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "door" || mode == "span" || mode == "stair" {
    let acted = agent(act_prompt(mode), #{
        label: "b9-span-act",
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
        label: "b9-span-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-span-skeptic",
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
            label: "b9-span-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    door_steps: g_door,
    span_steps: g_span,
    note: record_note,
});
