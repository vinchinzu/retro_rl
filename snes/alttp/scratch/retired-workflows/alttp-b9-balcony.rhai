let meta = #{
    name: "alttp-b9-balcony",
    description: "Recheck the Eastern 0xB9 layer-1 balcony. Walk only a strict legal path both counts agree on.",
    when_to_use: "After alttp-b9-layer. The 202-cell list was not checked by the skeptic, and the 105-step line steps on ledge and wall. Do not walk that line.",
    phases: [
        #{ title: "Count", detail: "two offline counts of the fixture, no ROM" },
        #{ title: "Act", detail: "stair replay, then a strict path or one predicted step" },
        #{ title: "Record", detail: "residual and the frontier sentence, then a skeptic" },
    ],
};

let count_schema = #{
    "type": "object",
    "required": ["strict_steps", "illegal_on_path", "entry_cells", "entry_stair_bytes", "candidate_col", "candidate_row", "candidate_byte", "note"],
    "properties": #{
        "strict_steps": #{ "type": "integer" },
        "illegal_on_path": #{ "type": "integer" },
        "entry_cells": #{ "type": "integer" },
        "entry_stair_bytes": #{ "type": "integer" },
        "candidate_col": #{ "type": "integer" },
        "candidate_row": #{ "type": "integer" },
        "candidate_byte": #{ "type": "integer" },
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

fn count_rules() {
    let p = "";
    p += "Use find_path from snes/alttp/dungeon/nav.py only. Read that function, room_origin, and tile_kind in snes/alttp/dungeon/tilemap.py before you write code. Walkable kinds are floor, shallow_water, door, stairs, and ice. Ledge, wall, and unknown are not walkable. Do not write a second search that allows them.\n";
    p += "Read snes/alttp/tests/fixtures/room_b9_layer1.json. Grids are 64 by 64, index [row][col]. Origin is room_origin of world (4808, 6006). Link poses are world pixels on the 8px phase find_path requires. Body is LINK_BODY from snes/alttp/walk_physics.py.\n";
    p += "Layer 0 reachable means 4-connected walkable cells from tile (31, 59), plus row 59 columns 31 and 32 even when the byte is 0x80. An entry cell is one of those whose same (row, col) on layer 1 is walkable. entry_cells is that count. Stair bytes are 0x1C, 0x1E, 0x1F, 0x3E, and 0x5E. entry_stair_bytes is how many entry cells have a stair byte on layer 0. A walkable layer-1 cell under a layer-0 cell is not a drop. The note must say how many of the entry cells have a stair byte on layer 1 as well.\n";
    p += "Start is the legal layer-1 pose on that phase within 16px of world (4808, 6006). If none, strict_steps is 0. Goal is the legal pose on that phase whose body covers door tile (31, 6) and whose distance to that tile is smallest. If no legal pose covers the door tile, strict_steps is 0. Search layer 1 only. Do not stitch the stair back to layer 0. Do not route through column 3.\n";
    p += "illegal_on_path is the number of tiles under LINK_BODY along the find_path result whose kind is not walkable. If that count is not 0, set strict_steps to 0. The 105-step line that returns through the stair and steps on ledge and wall is not a result. Do not rebuild it.\n";
    p += "candidate_byte is -1 unless one test fits. The landing component is the layer-1 cells a legal pose can occupy. The door component is the layer-1 walkable cells within 8 tiles of (31, 6), if that component does not touch the landing component. The cut is the set of attribute bytes on cells that touch both components. Set candidate_col, candidate_row, and candidate_byte only when that set has exactly one byte, tile_kind of that byte is neither wall nor ledge, and you pick the cut cell closest to the start. Otherwise all three are -1. Column 3 is never the candidate.\n";
    return p;
}

fn count_prompt(out_name) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Offline only. Do not boot a ROM. Do not run room_engine. Do not press A.\n";
    p += count_rules();
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Put the script in /tmp and delete it when you are done.\n";
    p += "Write only ";
    p += out_name;
    p += ". JSON keys: strict_steps, illegal_on_path, entry_cells, entry_stair_bytes, candidate_col, candidate_row, candidate_byte, path as a list of [x, y] world pixels (empty when strict_steps is 0), note. Do not edit maps, ORDER.md, PLAY.md, residual.md, PORT_PLAN.md, docs/STATUS.md, room_engine.py, primitives.py, or the fixture. Do not run bd. Do not git commit.\n";
    p += "Return those integers and note. strict_steps is 0 when find_path returns nothing. candidate_byte is -1 when there is no single candidate.\n";
    return p;
}

fn act_prompt(mode, col, row, byte) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". One emulator process. Read snes/alttp/scripts/pin_probe.py (open_pin, describe) and snes/alttp/ram.py read_snapshot before you step. Read retro_harness.env.write_state_bytes and state_path, and alttp.spine.tape.write_pin_provenance, before any save. Use the arguments those functions actually take.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy. Put the script in /tmp and delete it when you are done.\n";
    p += "Open Play_eog9_7. Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. If HP changes from 28, stop.\n";
    p += "Predicted stair: north on column 31 to row 54, west to column 25, north to (4808, 6056), still layer 0, room 0xB9. UP sets submodule 0x10. Control returns at (4808, 6006), layer 1, room 0xB9, module 0x07, submodule 0, HP 28/32, sword 1, lamp 1, keys 0, follower 0. Write that claim down before you hold UP. If room, layer, or xy is outside 8px of (4808, 6006), stop. Do not save. Return walked false.\n";
    p += "If the landing matches and Play_eog9_8.state is absent, save it beside Play_eog9_7.state. Provenance kind is play, source Play_eog9_7. If Play_eog9_8.state exists, do not overwrite it and do not pick another name. Do not overwrite Play_eog9_1 through Play_eog9_7.\n";
    if mode == "walk" {
        p += "Then read snes/alttp/docs/routes/full_clear/b9_path.json. Follow its path list. Halt at the first pose more than 8px off the next point, or if room or layer changes before the door. Entering room 0xA9 from the north door is success: record the glance and do not explore 0xA9. If the next pose would put LINK_BODY on a ledge, wall, or unknown tile, stop. Do not walk column 3. Do not walk back through the stair.\n";
    } else {
        p += "Then take one step of at most 8px from the landing toward tile column ";
        p += col;
        p += " row ";
        p += row;
        p += ", attribute byte ";
        p += byte;
        p += ". Write the predicted x, y, layer, and room before the step. Grade RAM after that one step. Do not take a second step. Do not save the probe pose over Play_eog9_8.\n";
    }
    p += "Return walked (true only if the landing matched and, for a walk, the path finished or the door entered 0xA9), saved (true only if Play_eog9_8.state was created this run), room (hex), layer (integer), and note (xy, hp, sword, lamp, keys, follower, and the grade).\n";
    return p;
}

fn record_prompt(mode, act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Mode is ";
    p += mode;
    p += ". Act note: ";
    p += act_note;
    p += "\n";
    p += "Read snes/alttp/docs/routes/full_clear/b9_path.json, snes/alttp/docs/routes/full_clear/b9_path_check.json, snes/alttp/docs/tasks/residual.md, snes/alttp/maps/room_b9.json, and the frontier paragraph of snes/alttp/docs/routes/full_clear/ORDER.md. Use read_file.\n";
    p += "The 105-step line is not body-legal. It steps on ledge and wall. Remove that claim from residual.md, from the last note in maps/room_b9.json, and from the frontier paragraph in ORDER.md. Replace it with the two counts and what the act note measured. If the two JSON files disagree, say the counts disagreed and do not pick a path. If mode is blocked, name the cut and say the north door still has no approach. If mode is walk and the act note entered room 0xA9, quote that glance and say the frontier is room 0xA9. If mode is probe, quote the predicted step and the grade.\n";
    p += "Do not mark any ORDER row after 4 as true. Do not add approachXy or a point whose label starts with north_. Do not press A. Do not claim a credits tape. Do not claim autoplay is finished. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, PLAY.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting only. Keep the Play_eog9_7 glance and the stair measurement. Return path residual.md and note (one sentence).\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read snes/alttp/docs/tasks/residual.md, snes/alttp/maps/room_b9.json, snes/alttp/docs/routes/full_clear/ORDER.md, snes/alttp/docs/routes/full_clear/b9_path.json, and snes/alttp/docs/routes/full_clear/b9_path_check.json with read_file.\n";
    p += "Count a problem if any of these are true: the 105-step line is still called body-legal or the route; a path through column 3 is offered as the walk; approachXy is set; a point label starts with north_; an ORDER row after 4 is true; a credits tape is claimed; autoplay is called finished; the text tells anyone to press A; AP-105 is marked done. Also count a problem if residual.md does not quote entry_cells from both JSON files. Quote the line for each problem.\n";
    p += "ok is true only when you read all five files and found none of those. Do not edit files. Do not boot a ROM.\n";
    p += "Return ok, problems (integer), and note (the quotes, or clean).\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. A skeptic rejected the write-up. The note is: ";
    p += problems;
    p += "\n";
    p += "Edit only snes/alttp/docs/tasks/residual.md, snes/alttp/maps/room_b9.json notes, and the frontier paragraph of snes/alttp/docs/routes/full_clear/ORDER.md. The 105-step line is not body-legal. Rows 1-4 are the only true rows. There is no approachXy and no north_ point. Do not press A. There is no credits tape. AP-105 stays open. Autoplay is not finished. Quote entry_cells from both b9_path.json and b9_path_check.json. Do not edit any other file. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note (what you changed).\n";
    return p;
}

phase("Count");
let counts = parallel([
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_path.json"),
        label: "b9-count",
        capability_mode: "execute",
        output_schema: count_schema,
    },
    #{
        prompt: count_prompt("snes/alttp/docs/routes/full_clear/b9_path_check.json"),
        label: "b9-count-check",
        capability_mode: "execute",
        output_schema: count_schema,
    },
]);

let g_steps = -1;
let s_steps = -1;
let g_illegal = -1;
let s_illegal = -1;
let g_byte = -1;
let s_byte = -1;
let g_col = -1;
let s_col = -1;
let g_row = -1;
let s_row = -1;
let g_note = "";
let s_note = "";
if counts.len() > 0 && counts[0] != () && counts[0].success && type_of(counts[0].output) == "map" {
    if counts[0].output.strict_steps != () { g_steps = counts[0].output.strict_steps; }
    if counts[0].output.illegal_on_path != () { g_illegal = counts[0].output.illegal_on_path; }
    if counts[0].output.candidate_byte != () { g_byte = counts[0].output.candidate_byte; }
    if counts[0].output.candidate_col != () { g_col = counts[0].output.candidate_col; }
    if counts[0].output.candidate_row != () { g_row = counts[0].output.candidate_row; }
    if counts[0].output.note != () { g_note = counts[0].output.note; }
}
if counts.len() > 1 && counts[1] != () && counts[1].success && type_of(counts[1].output) == "map" {
    if counts[1].output.strict_steps != () { s_steps = counts[1].output.strict_steps; }
    if counts[1].output.illegal_on_path != () { s_illegal = counts[1].output.illegal_on_path; }
    if counts[1].output.candidate_byte != () { s_byte = counts[1].output.candidate_byte; }
    if counts[1].output.candidate_col != () { s_col = counts[1].output.candidate_col; }
    if counts[1].output.candidate_row != () { s_row = counts[1].output.candidate_row; }
    if counts[1].output.note != () { s_note = counts[1].output.note; }
}
log(g_note);
log(s_note);

let mode = "unverified";
if g_steps >= 0 && s_steps >= 0 && g_steps == s_steps && g_illegal == 0 && s_illegal == 0 {
    if g_steps > 0 {
        mode = "walk";
    } else if g_byte > 0 && g_byte == s_byte && g_col == s_col && g_row == s_row && g_col >= 0 && g_row >= 0 {
        mode = "probe";
    } else {
        mode = "blocked";
    }
} else if g_steps >= 0 && s_steps >= 0 {
    mode = "disagree";
}
log(mode);

phase("Act");
let act_note = "no act";
if mode == "walk" || mode == "probe" {
    let acted = agent(act_prompt(mode, g_col.to_string(), g_row.to_string(), g_byte.to_string()), #{
        label: "b9-act",
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
        label: "b9-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-doc-skeptic",
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
            label: "b9-doc-fix",
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
