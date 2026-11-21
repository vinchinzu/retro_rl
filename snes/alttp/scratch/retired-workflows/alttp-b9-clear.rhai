let meta = #{
    name: "alttp-b9-clear",
    description: "Walk north from Play_eog9_11 without crossing the cannonball columns.",
    when_to_use: "After alttp-b9-aside. The grade held (4896, 5977). Do not walk x=4872. Do not step on 0x1E.",
    phases: [
        #{ title: "Count", detail: "two offline north paths that stay unexposed" },
        #{ title: "Act", detail: "open Play_eog9_11 and walk that path once" },
        #{ title: "Record", detail: "quote the pin and the grade" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["north_steps", "end_x", "end_y", "illegal_on_path", "exposed_poses", "note"],
    "properties": #{
        "north_steps": #{ "type": "integer" },
        "end_x": #{ "type": "integer" },
        "end_y": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "exposed_poses": #{ "type": "integer" },
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
    p += "Copy layer 1. Replace every stairs-kind byte with a legend wall byte, in the copy only. Do not edit nav.py, tilemap.py, or the fixture. Do not return the 29-step x=4872 line or the 3-step aside path as the result.\n";
    p += "Origin is room_origin of world (4896, 5977). A pose is exposed when LINK_BODY's x range intersects [4872, 4880) or [4888, 4896). Start is a legal pose within 8px of (4896, 5977). If that pose is exposed, north_steps is 0.\n";
    p += "On the masked grid, find the northernmost legal pose that can be reached by a find_path whose every pose is unexposed and whose y never increases. Ties go to the pose closest in x to 4896. north_steps is that length. If the end is not at least 16px north of the start, north_steps is 0. end_x and end_y are the end pose, or -1 when north_steps is 0. exposed_poses must be 0. illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. If it is not 0, north_steps is 0.\n";
    p += "Do not step onto byte 0x1E. tile_kind unknown is not a stair. The note must give the end pose and say whether any 0x1E cell touches it. Do not put 0x1E on the path.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it.\n";
    p += "Write only ";
    p += out_name;
    p += ". Keys: north_steps, end_x, end_y, illegal_on_path, exposed_poses, path (empty when north_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note.\n";
    return p;
}

fn act_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_11. Do not open Play_eog9_7 or Play_eog9_10. Do not replay the stair. Do not walk x=4872. Do not walk the 29-step line. Read pin_probe.open_pin and describe, and ram.read_snapshot, before you step. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_11. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_11. Do not hold UP on a 0x1E cell.\n";
    p += "Describe Play_eog9_11 first. A hit is room 0xB9, layer 1, x within 8 of 4896, y within 8 of 5977, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "A pose is exposed when LINK_BODY's x range intersects [4872, 4880) or [4888, 4896). Read the path in snes/alttp/docs/routes/full_clear/b9_clear.json. If the first point is more than 8px from the live pose, stop. If any point is exposed or has y greater than the pin y, stop before moving. Follow the list. Halt when a pose is more than 8px off the next point, when y increases, when the body would be exposed, when room or layer changes, or when HP changes.\n";
    p += "Save Play_eog9_12 only when the file is absent, the last pose matched within 8px, HP is 28, room is 0xB9, layer is 1, and the live y is at least 16px north of 5977.\n";
    p += "Report cannonball slots. Slot 15 was on x=4872. Do not fight.\n";
    p += "Return walked (true only when the last pose matched and y is at least 16px north), saved (true only if Play_eog9_12.state was created), room, layer, and note (the pin glance, xy, hp, sword, lamp, keys, follower, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_clear.json, b9_clear_check.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file. Also read Play_eog9_11.provenance.json under snes/alttp/custom_integrations/Zelda3-Snes.\n";
    p += "Play_eog9_11 exists. Kind resumed, source_run Play_eog9_10, frame 16. The grade held (4896, 5977), layer 1, room 0xB9, HP 28. The integers aside_steps 3 and end_x 4896 agreed, and that path was walked. Do not write that the path was not picked. The pose is not a door approach. The 29-step x=4872 line stays withdrawn. Slot 15 met Link at (4872, 5947). Byte 0x1E is unknown, not a stair.\n";
    p += "Quote north_steps, end_x, and end_y from both clear files. Quote the act note. If those integers disagree, do not pick a path.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, Play_eog9_10, and Play_eog9_11. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_clear.json, and b9_clear_check.json under snes/alttp.\n";
    p += "Count a problem if the 29-step line is offered as the walk, if x=4872 north is called safe, if Play_eog9_11 is missing, if north_steps from both clear files are not quoted, if (4896, 5977) is called a door approach, if the aside path is called not picked, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all six files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. The 29-step line is withdrawn. x=4872 north is not safe. Play_eog9_11 held (4896, 5977) and that path was picked. It is not a door approach. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote north_steps from both clear files. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_clear.json"),
        label: "b9-clear",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_clear_check.json"),
        label: "b9-clear-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_steps = -1;
let s_steps = -1;
let g_x = -9999;
let s_x = -9999;
let g_y = -9999;
let s_y = -9999;
let g_illegal = -1;
let s_illegal = -1;
let g_exposed = -1;
let s_exposed = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.north_steps != () { g_steps = counts[0].output.north_steps; }
    if counts[0].output.end_x != () { g_x = counts[0].output.end_x; }
    if counts[0].output.end_y != () { g_y = counts[0].output.end_y; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.exposed_poses != () { g_exposed = counts[0].output.exposed_poses; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.north_steps != () { s_steps = counts[1].output.north_steps; }
    if counts[1].output.end_x != () { s_x = counts[1].output.end_x; }
    if counts[1].output.end_y != () { s_y = counts[1].output.end_y; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.exposed_poses != () { s_exposed = counts[1].output.exposed_poses; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_steps >= 0 && g_steps == s_steps && g_x == s_x && g_y == s_y && g_illegal == 0 && s_illegal == 0 && g_exposed == 0 && s_exposed == 0 {
    if g_steps > 0 {
        mode = "clear";
    } else {
        mode = "blocked";
    }
} else if g_steps >= 0 && s_steps >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "clear" {
    let acted = agent(act_prompt(), #{
        label: "b9-clear-act",
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
        label: "b9-clear-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-clear-skeptic",
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
            label: "b9-clear-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    north_steps: g_steps,
    end_x: g_x,
    end_y: g_y,
    note: record_note,
});
