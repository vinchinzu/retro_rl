let meta = #{
    name: "alttp-b9-onestep",
    description: "One north step from Play_eog9_10 inside the opening clear window. Do not stand still.",
    when_to_use: "After alttp-b9-gap. Frames 0-26 were clear at (4872, 5977). The 82-frame gap is after knockback. Do not hold for 240 frames.",
    phases: [
        #{ title: "Step", detail: "one 8px north step if the opening gap is still there" },
        #{ title: "Record", detail: "quote the grade and do not call the 82-frame gap a route" },
    ],
};

let act_schema = #{
    "type": "object",
    "required": ["stepped", "saved", "hp", "note"],
    "properties": #{
        "stepped": #{ "type": "boolean" },
        "saved": #{ "type": "boolean" },
        "hp": #{ "type": "integer" },
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

fn act_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_10. Do not open Play_eog9_11. Do not replay the stair. Do not hold still for 240 frames. Do not walk the 29-step line. Do not walk east. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_10. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_11.\n";
    p += "Describe Play_eog9_10 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5977, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. The measured open is slot 15 at about (4872, 5912), moving south about 1.5px per frame. Frames 0-26 of b9_balls.json were clear at link y=5977. The 82-frame gap in that file is after knockback to y=6016 and is not a window. Do not use it.\n";
    p += "A same-column ball has abs(x - 4872) <= 8. If any same-column ball already has abs(y - link_y) <= 24, do not step. If the nearest same-column ball is at least 48px north of link y, write this prediction before input: one north step of at most 8px, x stays within 8 of 4872, y decreases, HP stays 28, room 0xB9, layer 1. Take that one step immediately. Grade it. If HP changes, stop. Do not take a second step.\n";
    p += "Save Play_eog9_12 only when the file is absent, the step matched within 8px, HP is 28, y is at least 8px north of the pin, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, nearest ball xy, prediction, grade, and final xy).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_balls.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "b9_balls.json gap_frames 82 is the longest clear run after Link was already knocked from (4872, 5977) to (4872, 6016). HP fell from 28 to 16 during the 240-frame hold. That gap is not a route. The opening clear run at the pin was frames 0-26. The step in that run was not taken. Play_eog9_12 was not saved. The 29-step x=4872 line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If the one step missed or HP fell, say the xy and do not call it a route. If it held, quote the saved pin and say it is not the door.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, Play_eog9_10, and Play_eog9_11. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if more than one north step is claimed, if a pin was saved after an HP drop, if (4896, 5977) is called a door approach, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. The 82-frame gap is not a route. The 29-step line is withdrawn. One north step at most. No save after an HP drop. (4896, 5977) is not a door approach. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-onestep-act",
    capability_mode: "all",
    output_schema: act_schema,
});
let act_note = "no act";
let stepped = false;
if acted != () && acted.success && type_of(acted.output) == "map" {
    if acted.output.note != () { act_note = acted.output.note; }
    if acted.output.stepped != () { stepped = acted.output.stepped; }
}
log(act_note);

phase("Record");
let record_note = "no record";
if act_note != "no act" {
    let recorded = agent(record_prompt(act_note), #{
        label: "b9-onestep-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-onestep-skeptic",
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
            label: "b9-onestep-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
