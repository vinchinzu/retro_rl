let meta = #{
    name: "alttp-b9-layer",
    description: "Dump Eastern hall layers, test a layer change offline, and write the play plan",
    when_to_use: "After the false north_002 polyline was deleted. Frontier is the layer change in room 0xB9. Do not walk it.",
    phases: [
        #{ title: "Measure", detail: "one pin load, both attribute grids, no buttons" },
        #{ title: "Geometry", detail: "offline entry cells, no room_engine" },
        #{ title: "Audit", detail: "six lanes against the 227-row order" },
        #{ title: "Write", detail: "play plan, then one skeptic" },
    ],
};

let measure_schema = #{
    "type": "object",
    "required": ["wrote", "room", "layer", "note"],
    "properties": #{
        "wrote": #{ "type": "boolean" },
        "room": #{ "type": "string" },
        "layer": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

let geo_schema = #{
    "type": "object",
    "required": ["entry_cells", "frontier", "note"],
    "properties": #{
        "entry_cells": #{ "type": "integer" },
        "frontier": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
};

let audit_schema = #{
    "type": "object",
    "required": ["ok", "false_claims", "note"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "false_claims": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

let play_schema = #{
    "type": "object",
    "required": ["path", "items", "frontier", "note"],
    "properties": #{
        "path": #{ "type": "string" },
        "items": #{ "type": "integer" },
        "frontier": #{ "type": "string" },
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

fn measure_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You dump both indoor attribute layers of Play_eog9_7. One emulator process. No buttons.\n";
    p += "Read snes/alttp/scripts/pin_probe.py (open_pin, describe) and snes/alttp/dungeon/tilemap.py (read_room_tiles) and snes/alttp/ram.py (read_snapshot) before you write code. Use the field names those functions actually return.\n";
    p += "From the repo root, PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy. Put the script in /tmp and delete it when you are done. Do not commit it.\n";
    p += "Call pin_probe.open_pin('Play_eog9_7'). Read RAM once. Call read_room_tiles for layer 0 and layer 1. Print describe(). Write snes/alttp/tests/fixtures/room_b9_layer1.json with room, link (x, y, layer, hp), layer0 and layer1 as 64 by 64 integer arrays. Close the env. Do not pass --press. Do not step. Do not save a state. Do not call room_engine. Do not overwrite tests/fixtures/room_b9.json or tests/fixtures/room_61.json. Do not edit maps/room_b9.json.\n";
    p += "Do not edit PORT_PLAN.md, docs/STATUS.md, ORDER.md, residual.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit. Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A.\n";
    p += "Return wrote (true only if the new json exists and both grids are 64 by 64), room (hex from the snapshot), layer (Link's layer integer), and note (xy, hp, and whether the pin is still room 0xB9).\n";
    return p;
}

fn geo_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You test whether room 0xB9 has a layer change into the north door. Offline only. Do not boot a ROM. Do not run room_engine. Do not press A.\n";
    p += "Read snes/alttp/tests/fixtures/room_b9_layer1.json and snes/alttp/tests/fixtures/room_b9.json and snes/alttp/dungeon/nav.py find_path and LINK_BODY. Read maps/room_b9.json. The north door has no approachXy. south_landing (4856, 6104) is the only point. Do not add points. Do not set approachXy. Do not restore north_001 through north_099.\n";
    p += "If room_b9_layer1.json is missing or either grid is not 64 by 64, append one note to maps/room_b9.json and replace the sentence 'The layer-1 grid is not in the fixture yet.' in docs/routes/full_clear/ORDER.md with 'Layer-1 dump missing. Frontier stays the layer change.' Return entry_cells 0, frontier 'layer change', and that note. Do not invent cells.\n";
    p += "Otherwise run Python with PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and uv run. Use find_path(tiles, start_px, goal_px, LINK_BODY). Goal pixel is tile (31,6) times 8. Walkable kinds are floor, shallow_water, door, stairs, ice. Byte 0x80 is unknown and is not walkable.\n";
    p += "Layer 0 reachable means 4-connected walkable cells from floor tile (58, 31), plus the landing cells row 59 cols 31-32 where Link stands even though they are byte 0x80. An entry cell is one of those layer-0 cells whose same (row, col) on layer 1 is walkable and whose layer-1 find_path reaches the goal. Also test the noted drops: cols 25 and 43 on the south floor, and row 10 cols 25 and 43, but count one only when it is inside the layer-0 reachable set.\n";
    p += "Append one sentence to the notes array in maps/room_b9.json. If entry_cells is 0, say no layer-0 stand reaches a layer-1 path to tile (31,6), and do not run room_engine. If entry_cells is greater than 0, list the tile coordinates only. Still do not add points, do not set approachXy, and do not run room_engine.\n";
    p += "In ORDER.md replace only the sentence 'The layer-1 grid is not in the fixture yet.' with the same measurement. Do not mark any row true. Do not change rows 1-4. Do not edit PORT_PLAN.md, docs/STATUS.md, residual.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Return entry_cells (integer), frontier (one short string: 'layer change' or 'entry tiles <list>'), and note (the sentence you wrote).\n";
    return p;
}

fn audit_prompt(label, section, census, out_file) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are audit lane ";
    p += label;
    p += ". Read docs/routes/full_clear/ORDER.md with read_file. Read these census files if they exist: ";
    p += census;
    p += ".\n";
    p += "Your section is: ";
    p += section;
    p += ".\n";
    p += "A false claim is any of these, quoted from the file: a status cell containing the word true on a row other than 1, 2, 3, or 4; a walk called runnable after the frontier; a seven-crystal pyramid gate that the Pyramid gate section says the pin does not encode; a credits tape claimed to exist; north_002 named as the current path or approach. The cannonball row must say not to press A.\n";
    p += "Write only ";
    p += out_file;
    p += ". If you find none, the file body is the single word NONE. Otherwise one paragraph per claim, with the quoted line.\n";
    p += "Do not edit ORDER.md, maps, PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return ok (true only when the file body is NONE), false_claims (integer count), and note (one sentence).\n";
    return p;
}

fn play_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Write snes/alttp/docs/routes/full_clear/PLAY.md. Read ORDER.md, maps/room_b9.json, docs/tasks/residual.md, and every audit_*.md under docs/routes/full_clear/. Use read_file. Do not answer from memory.\n";
    p += "PLAY.md has four sections. Runnable prefix: Clean tape recordings/clean_escape_20261002 ends in room 0x12, and Play_eog9_7 is room 0xB9 at (4856, 6104), sword 1, lamp 1, HP 28/32, follower 0, keys 0. Rows 1-4 of ORDER are the only true rows. Frontier: quote the layer-change sentence now in ORDER and the map notes, including the entry-cell count if the geometry note is there. Logic order: 227 census rows through Ganon's Triforce, which has an empty requires list. The pin does not encode seven crystals on the pyramid. No row after 4 is playable. Credits: AP-197 is not implemented and there is no credits tape. The plan ends at the Triforce, then that gap.\n";
    p += "Add a Contradictions section. Quote each audit file that is not NONE. If an audit file is missing, say that lane is unverified.\n";
    p += "Do not claim autoplay is finished. Do not add approach points. Do not edit maps, ORDER.md, PORT_PLAN.md, docs/STATUS.md, residual.md, room_engine.py, or primitives.py. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return path, items 227, frontier (the frontier sentence), and note (one sentence on whether any audit found a false claim).\n";
    return p;
}

fn skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read snes/alttp/docs/routes/full_clear/PLAY.md, snes/alttp/docs/routes/full_clear/ORDER.md, and snes/alttp/maps/room_b9.json with read_file.\n";
    p += "Set ok false, and count a problem, if any of these are true: PLAY.md says the north door was walked or that autoplay reaches credits; ORDER.md marks a row after 4 as true; ORDER.md still names north_002 as the frontier; maps/room_b9.json has approachXy or a point whose label starts with north_; the cannonball text tells anyone to press A. Quote the line for each problem.\n";
    p += "ok is true only when you read all three files and found none of those. Do not edit files. Do not boot a ROM.\n";
    p += "Return ok, problems (integer), and note (the quotes, or 'clean').\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. A skeptic rejected PLAY.md. The note is: ";
    p += problems;
    p += "\n";
    p += "Edit only snes/alttp/docs/routes/full_clear/PLAY.md so it matches ORDER.md and maps/room_b9.json. Rows 1-4 are the only true rows. The frontier is the layer change, not north_002. There is no approachXy. Do not press A on the cannonball chest. There is no credits tape. Do not edit any other file. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return path 'snes/alttp/docs/routes/full_clear/PLAY.md', items 227, frontier (one sentence), and note (what you changed).\n";
    return p;
}

phase("Measure");
let measured = agent(measure_prompt(), #{
    label: "b9-layers",
    capability_mode: "all",
    output_schema: measure_schema,
});
let measured_note = "no measure";
if measured != () && measured.success && measured.output.note != () {
    measured_note = measured.output.note;
}
log(measured_note);

phase("Geometry");
let geo = agent(geo_prompt(), #{
    label: "b9-geometry",
    capability_mode: "execute",
    output_schema: geo_schema,
});
let frontier = "layer change";
if geo != () && geo.success && geo.output.frontier != () {
    frontier = geo.output.frontier;
}
log(frontier);

phase("Audit");
let audits = parallel([
    #{
        prompt: audit_prompt(
            "eastern",
            "Eastern Palace rows and the frontier paragraph",
            "snes/alttp/docs/routes/full_clear/dungeons_early.md",
            "snes/alttp/docs/routes/full_clear/audit_eastern.md"
        ),
        label: "audit-eastern",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
    #{
        prompt: audit_prompt(
            "light",
            "Book, boots, Desert, and Light Death Mountain",
            "snes/alttp/docs/routes/full_clear/lightworld_main.md snes/alttp/docs/routes/full_clear/lightworld_field.md snes/alttp/docs/routes/full_clear/lightworld_mountain.md",
            "snes/alttp/docs/routes/full_clear/audit_light.md"
        ),
        label: "audit-light",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
    #{
        prompt: audit_prompt(
            "dark",
            "Dark World field and Dark Death Mountain",
            "snes/alttp/docs/routes/full_clear/darkworld_field.md snes/alttp/docs/routes/full_clear/darkworld_mountain.md",
            "snes/alttp/docs/routes/full_clear/audit_dark.md"
        ),
        label: "audit-dark",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
    #{
        prompt: audit_prompt(
            "early",
            "Hera, Tower, and the early dungeon rows",
            "snes/alttp/docs/routes/full_clear/dungeons_early.md snes/alttp/docs/routes/full_clear/dungeons_mid.md",
            "snes/alttp/docs/routes/full_clear/audit_early.md"
        ),
        label: "audit-early",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
    #{
        prompt: audit_prompt(
            "late",
            "Ice, Mire, Turtle Rock, Ganon's Tower item rows",
            "snes/alttp/docs/routes/full_clear/dungeons_late.md",
            "snes/alttp/docs/routes/full_clear/audit_late.md"
        ),
        label: "audit-late",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
    #{
        prompt: audit_prompt(
            "finale",
            "Triforce row 227, the Pyramid gate section, and the credits paragraph",
            "snes/alttp/docs/routes/full_clear/dungeons_late.md",
            "snes/alttp/docs/routes/full_clear/audit_finale.md"
        ),
        label: "audit-finale",
        capability_mode: "read-write",
        output_schema: audit_schema,
    },
]);

let claims = 0;
for row in audits {
    if row != () && row.success && row.output.false_claims != () {
        claims += row.output.false_claims;
    }
}
log("false_claims " + claims.to_string());

phase("Write");
let play = agent(play_prompt(), #{
    label: "play-plan",
    capability_mode: "read-write",
    output_schema: play_schema,
});
let skeptic = agent(skeptic_prompt(), #{
    label: "play-skeptic",
    capability_mode: "read-only",
    output_schema: skeptic_schema,
});
let skeptic_note = "";
let problems = 0;
if skeptic != () && skeptic.success && skeptic.output.note != () {
    skeptic_note = skeptic.output.note;
}
if skeptic != () && skeptic.success && skeptic.output.problems != () {
    problems = skeptic.output.problems;
}
if problems > 0 {
    agent(fix_prompt(skeptic_note), #{
        label: "play-fix",
        capability_mode: "read-write",
        output_schema: play_schema,
    });
}

let play_note = "";
if play != () && play.success && play.output.note != () {
    play_note = play.output.note;
}
complete(#{
    frontier: frontier,
    items: 227,
    false_claims: claims,
    skeptic_problems: problems,
    note: play_note,
});
