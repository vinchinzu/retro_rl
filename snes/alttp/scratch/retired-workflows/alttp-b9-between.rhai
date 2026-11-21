let meta = #{
    name: "alttp-b9-between",
    description: "One east step from Play_eog9_12, then hold until the west cannonball is south. Do not step north.",
    when_to_use: "After alttp-b9-onestep. Play_eog9_12 is (4872, 5969), HP 28. The body cannot stand off both ball columns on this row. Do not take another north step.",
    phases: [
        #{ title: "Step", detail: "one 8px east step, then hold for the west ball to pass" },
        #{ title: "Record", detail: "quote the grade and do not call it the door" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_12. Do not open Play_eog9_10 or Play_eog9_11. Do not replay the stair. Do not step north. Do not walk the 29-step line. Do not hold for 240 frames. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_12. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_12.\n";
    p += "Describe Play_eog9_12 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5969, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. A west-column ball has abs(x - 4872) <= 8. If any west-column ball already has abs(y - link_y) <= 24, do not step. If the nearest west-column ball is at least 40px north of link y, write this prediction before input: one east step of at most 8px, y stays within 8 of the pin, x increases, HP stays 28, room 0xB9, layer 1. Take that one step immediately. Do not take a second step. Do not press UP.\n";
    p += "After the step, hold still. Stop the hold when that west-column ball has y greater than link_y + 24, or when 80 frames have passed since the open, or when HP changes. If HP changes, stop and do not save.\n";
    p += "Save Play_eog9_13 only when the file is absent, the east step matched within 8px, x is at least 8px east of the pin, y is within 8 of the pin, HP is 28, the west-column ball is south of link by more than 24px, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, grade, frames held, final xy, and final ball xy).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Play_eog9_12 is (4872, 5969), layer 1, room 0xB9, HP 28. It is not the door. The 82-frame gap after the knock to (4872, 6016) is not a route. The 29-step x=4872 line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If the east step missed or HP fell, say the xy and do not call it a route. If it held, quote the saved pin and say it is not the door.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, Play_eog9_10, Play_eog9_11, and Play_eog9_12. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if a north step from Play_eog9_12 is claimed, if a pin was saved after an HP drop, if (4896, 5977) is called a door approach, if Play_eog9_12 or Play_eog9_13 is called the door, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. The 82-frame gap is not a route. The 29-step line is withdrawn. No north step from Play_eog9_12. No save after an HP drop. (4896, 5977) is not a door approach. Play_eog9_12 is not the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-between-act",
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
        label: "b9-between-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-between-skeptic",
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
            label: "b9-between-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
