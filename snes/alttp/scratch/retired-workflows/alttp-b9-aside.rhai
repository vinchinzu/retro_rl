let meta = #{
    name: "alttp-b9-aside",
    description: "Step off the Eastern cannonball column. Do not walk north on x=4872.",
    when_to_use: "After alttp-b9-span. Slot 15 met Link at (4872, 5947) and HP fell to 24. The 29-step line is withdrawn.",
    phases: [
        #{ title: "Count", detail: "two offline sidesteps off columns 33 and 35" },
        #{ title: "Act", detail: "open Play_eog9_10 and walk only that sidestep" },
        #{ title: "Record", detail: "withdraw the column-33 span" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["aside_steps", "end_x", "illegal_on_path", "goes_north", "note"],
    "properties": #{
        "aside_steps": #{ "type": "integer" },
        "end_x": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "goes_north": #{ "type": "integer" },
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
    p += "Copy layer 1. Replace every stairs-kind byte with a legend wall byte, in the copy only. Do not edit nav.py, tilemap.py, or the fixture. Do not return the 29-step path in b9_span.json. That line stays on x=4872 and is withdrawn.\n";
    p += "Origin is room_origin of world (4872, 5977). A pose is exposed when LINK_BODY's x range intersects [4872, 4880) or [4888, 4896). Those are the cannonball columns. Start is a legal pose within 8px of (4872, 5977). It is exposed. That is allowed only for the start.\n";
    p += "Search the masked grid for the shortest find_path to a legal pose that is not exposed, whose y is within 8px of the start y, and whose x is at least 16px from 4872. No pose may have y less than the start y. Going north is forbidden. goes_north is 1 if any considered pose has y less than the start y, and aside_steps is then 0. If both an east end and a west end exist, keep the shorter. end_x is that pose's x, or -1 when aside_steps is 0.\n";
    p += "illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. If it is not 0, aside_steps is 0. The note must say whether the end is east or west, and list the unmasked row-14 bytes 0x1E with their columns. tile_kind unknown is not a stair. Do not put those cells on the path.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it.\n";
    p += "Write only ";
    p += out_name;
    p += ". Keys: aside_steps, end_x, illegal_on_path, goes_north, path (empty when aside_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note.\n";
    return p;
}

fn act_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_10. Do not open Play_eog9_7. Do not replay the stair. Do not walk the 29-step line. Do not walk north. Read pin_probe.open_pin and describe, and ram.read_snapshot, before you step. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_10. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_10.\n";
    p += "Describe Play_eog9_10 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5977, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the path in snes/alttp/docs/routes/full_clear/b9_aside.json. If the first point is more than 8px from the live pose, stop. Follow the list. Halt when a pose is more than 8px off the next point, when y is more than 8px north of the pin, when room or layer changes, or when HP changes. Do not hold UP. Do not press the stair.\n";
    p += "Save Play_eog9_11 only when the file is absent, the last pose matched within 8px, HP is 28, room is 0xB9, layer is 1, and the live x is at least 16px from 4872.\n";
    p += "Report cannonball slots if they are on screen. Slot 15 was at (4872, 5945) when HP fell. Do not fight.\n";
    p += "Return walked (true only when the last pose matched), saved (true only if Play_eog9_11.state was created), room, layer, and note (the pin glance, xy, hp, sword, lamp, keys, follower, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_aside.json, b9_aside_check.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "The 29-step span is withdrawn. Both span files agreed on span_steps 29 and edge_row 15, and the walk followed x=4872. Slot 15 cannonball met Link at (4872, 5947). HP fell from 28 to 24. No pin was saved. Play_eog9_10 remains the pose before that walk: (4872, 5977), layer 1, room 0xB9, HP 28, kind resumed, source Play_eog9_9, frame 5. That pose is not a door approach.\n";
    p += "Quote aside_steps and end_x from both aside files. Quote the act note. If the files disagree, do not pick a path. Row 14 bytes 0x1E are unknown, not stairs, and are not a probe.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, and Play_eog9_10. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_aside.json, and b9_aside_check.json under snes/alttp.\n";
    p += "Count a problem if the 29-step line is still offered as the walk, if x=4872 north is called safe, if (4872, 5977) is called a door approach, if aside_steps from both aside files are not quoted, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all six files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. The 29-step line is withdrawn. x=4872 north is not safe. (4872, 5977) is not a door approach. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote aside_steps from both aside files. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_aside.json"),
        label: "b9-aside",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_aside_check.json"),
        label: "b9-aside-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_steps = -1;
let s_steps = -1;
let g_x = -9999;
let s_x = -9999;
let g_illegal = -1;
let s_illegal = -1;
let g_north = -1;
let s_north = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.aside_steps != () { g_steps = counts[0].output.aside_steps; }
    if counts[0].output.end_x != () { g_x = counts[0].output.end_x; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.goes_north != () { g_north = counts[0].output.goes_north; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.aside_steps != () { s_steps = counts[1].output.aside_steps; }
    if counts[1].output.end_x != () { s_x = counts[1].output.end_x; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.goes_north != () { s_north = counts[1].output.goes_north; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_steps >= 0 && g_steps == s_steps && g_x == s_x && g_illegal == 0 && s_illegal == 0 && g_north == 0 && s_north == 0 {
    if g_steps > 0 {
        mode = "aside";
    } else {
        mode = "blocked";
    }
} else if g_steps >= 0 && s_steps >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "aside" {
    let acted = agent(act_prompt(), #{
        label: "b9-aside-act",
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
        label: "b9-aside-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-aside-skeptic",
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
            label: "b9-aside-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    aside_steps: g_steps,
    end_x: g_x,
    note: record_note,
});
