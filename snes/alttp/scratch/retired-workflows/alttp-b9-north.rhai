let meta = #{
    name: "alttp-b9-north",
    description: "North-only layer-1 path from the Eastern 0xB9 landing. Stair tiles are transitions, not floor.",
    when_to_use: "After alttp-b9-balcony. The 109-step lists leave the landing by increasing y. Do not walk them.",
    phases: [
        #{ title: "Count", detail: "two offline north-only counts, stairs masked" },
        #{ title: "Act", detail: "stair replay, then the north path or one north step" },
        #{ title: "Record", detail: "replace the stale 105-step sentence" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["strict_steps", "illegal_on_path", "goes_south", "note"],
    "properties": #{
        "strict_steps": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "goes_south": #{ "type": "integer" },
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
    p += "Read snes/alttp/dungeon/nav.py find_path and room_origin, snes/alttp/dungeon/tilemap.py tile_kind, and snes/alttp/walk_physics.py LINK_BODY before you write code. Read snes/alttp/tests/fixtures/room_b9_layer1.json. Grids are 64 by 64, index [row][col].\n";
    p += "Do not return the path in snes/alttp/docs/routes/full_clear/b9_path.json or b9_path_check.json. Both leave the landing by increasing y. That is south, back onto the stair. goes_south is 1 for any result that contains a pose with y greater than its start y, and strict_steps is then 0.\n";
    p += "Copy the layer-1 grid. find_path treats stairs as walkable. In the copy only, replace every byte whose tile_kind is stairs with a byte whose tile_kind is wall. Read the legend and use one of those wall bytes. Do not edit nav.py, tilemap.py, or the fixture.\n";
    p += "Origin is room_origin of world (4808, 6009). Poses are world pixels on the 8px phase find_path requires. Start is a legal pose on the masked grid within 8px of (4808, 6009). If none, strict_steps is 0. Goal is the legal pose whose body covers door tile (31, 6) and whose distance to that tile is smallest. Search the masked layer 1 only. Do not stitch layer 0.\n";
    p += "illegal_on_path counts tiles under LINK_BODY whose kind is not floor, shallow_water, door, or ice. Stairs, ledge, wall, and unknown count as illegal. If illegal_on_path is not 0, strict_steps is 0. If any pose has y greater than the start y, goes_south is 1 and strict_steps is 0. Otherwise goes_south is 0.\n";
    p += "When strict_steps is 0, the note must list the attribute byte and tile_kind on the masked grid for the cells one body-north of the start, columns 20 through 40.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it when you are done.\n";
    p += "Write only ";
    p += out_name;
    p += ". JSON keys: strict_steps, illegal_on_path, goes_south, path as a list of [x, y] (empty when strict_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return strict_steps, illegal_on_path, goes_south, and note.\n";
    return p;
}

fn act_prompt(mode) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". One emulator process. Read snes/alttp/scripts/pin_probe.py open_pin and describe, and snes/alttp/ram.py read_snapshot, before you step. Read retro_harness.env.write_state_bytes and state_path, and alttp.spine.tape.write_pin_provenance, before any save. Kind must be one of natural, resumed, poke, whatif, or scan. Kind play is rejected. Use kind resumed, source_run Play_eog9_7, and frame the step count.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy. Put the script in /tmp and delete it when you are done.\n";
    p += "Open Play_eog9_7. Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop.\n";
    p += "Stair claim, written before you hold UP: north on column 31 to row 54, west to column 25, north to (4808, 6056), still layer 0, room 0xB9. UP sets submodule 0x10. Control returns in room 0xB9, layer 1, x within 8px of 4808, y within 8px of 6009. HP 28/32, sword 1, lamp 1, keys 0, follower 0. y=6006 and y=6009 are both hits. Do not fail the grade for a 3px gap inside that band. If room or layer misses, or x or y is outside that band, stop and do not save.\n";
    p += "If the landing matches and Play_eog9_8.state is absent, save it beside Play_eog9_7.state with kind resumed. If Play_eog9_8.state exists, do not overwrite it. Do not overwrite Play_eog9_1 through Play_eog9_7.\n";
    if mode == "walk" {
        p += "Then read snes/alttp/docs/routes/full_clear/b9_north.json. If its first point is more than 8px from the live landing, stop. Do not walk south to meet it. Follow the list. Halt at the first pose more than 8px off the next point, or if y becomes greater than the landing y, or if room or layer changes before the door. Entering room 0xA9 is success: record the glance and do not explore 0xA9. If the next pose would put LINK_BODY on stairs, ledge, wall, or unknown, stop.\n";
    } else {
        p += "Then take one step of at most 8px north from the landing. Read the layer-1 byte under that body cell from the fixture before you step, and write the predicted x, y, layer, and room. If that byte's tile_kind is wall, ledge, or unknown, predict that xy stays inside 8px of the landing. Grade RAM after that one step. Do not take a second step. Do not walk south. Do not save the probe pose over Play_eog9_8.\n";
    }
    p += "Return walked (true only if the landing matched and, for a walk, the path finished or the door entered 0xA9), saved (true only if Play_eog9_8.state was created this run), room (hex), layer (integer), and note (xy, hp, sword, lamp, keys, follower, prediction, and grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read snes/alttp/docs/routes/full_clear/b9_north.json, b9_north_check.json, snes/alttp/docs/tasks/residual.md, snes/alttp/maps/room_b9.json, snes/alttp/docs/routes/full_clear/ORDER.md, and snes/alttp/docs/routes/full_clear/PLAY.md. Use read_file.\n";
    p += "The 105-step line is not body-legal. The 109-step lists in b9_path.json and b9_path_check.json leave the landing by increasing y. They are not the route. Replace those claims in residual.md, the last note of maps/room_b9.json, the frontier paragraph of ORDER.md, and the PLAY.md sentence that still calls 105 steps the shortest body-legal path.\n";
    p += "Quote strict_steps and goes_south from both north JSON files. Quote the act note. If mode is walk and the act note entered room 0xA9, say the frontier is room 0xA9. If mode is probe, quote the prediction and the grade. If the two JSON files disagree, say so and do not pick a path.\n";
    p += "Do not mark any ORDER row after 4 as true. Do not add approachXy or a point whose label starts with north_. Do not press A. Do not claim a credits tape. Do not claim autoplay is finished. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting only. Keep the Play_eog9_7 glance and the stair measurement, including the live landing (4808, 6009). Return path residual.md and note (one sentence).\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, b9_north.json, and b9_north_check.json under snes/alttp. Use read_file.\n";
    p += "Count a problem if any of these are true: 105 steps is still called body-legal or the route; a 109-step list is offered as the walk; a path that increases y from the landing is offered as the walk; approachXy is set; a point label starts with north_; an ORDER row after 4 is true; a credits tape is claimed; autoplay is called finished; the text tells anyone to press A; AP-105 is marked done. Also count a problem if residual.md does not quote strict_steps from both north JSON files. Quote the line for each problem.\n";
    p += "ok is true only when you read all six files and found none of those. Do not edit files. Do not boot a ROM.\n";
    p += "Return ok, problems (integer), and note (the quotes, or clean).\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. A skeptic rejected the write-up. The note is: ";
    p += problems;
    p += "\n";
    p += "Edit only snes/alttp/docs/tasks/residual.md, the notes in snes/alttp/maps/room_b9.json, the frontier paragraph of snes/alttp/docs/routes/full_clear/ORDER.md, and the stale path sentence in snes/alttp/docs/routes/full_clear/PLAY.md. The 105-step line is not body-legal. The 109-step lists go south and are not the route. Rows 1-4 are the only true rows. There is no approachXy and no north_ point. Do not press A. There is no credits tape. AP-105 stays open. Autoplay is not finished. Quote strict_steps from both b9_north JSON files. Do not edit any other file. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note (what you changed).\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_north.json"),
        label: "b9-north",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_north_check.json"),
        label: "b9-north-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_steps = -1;
let s_steps = -1;
let g_illegal = -1;
let s_illegal = -1;
let g_south = -1;
let s_south = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.strict_steps != () { g_steps = counts[0].output.strict_steps; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.goes_south != () { g_south = counts[0].output.goes_south; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.strict_steps != () { s_steps = counts[1].output.strict_steps; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.goes_south != () { s_south = counts[1].output.goes_south; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_steps >= 0 && s_steps >= 0 && g_steps == s_steps && g_illegal == 0 && s_illegal == 0 && g_south == 0 && s_south == 0 {
    if g_steps > 0 {
        mode = "walk";
    } else {
        mode = "probe";
    }
} else if g_steps >= 0 && s_steps >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "walk" || mode == "probe" {
    let acted = agent(act_prompt(mode), #{
        label: "b9-north-act",
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
        label: "b9-north-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-north-skeptic",
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
            label: "b9-north-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    mode: mode,
    strict_steps: g_steps,
    check_steps: s_steps,
    note: record_note,
});
