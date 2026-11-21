let meta = #{
    name: "alttp-b9-back",
    description: "Step west from Play_eog9_22 onto x=4888, then take up to four north steps.",
    when_to_use: "After alttp-b9-deeper. Play_eog9_22 is (4904, 5886), HP 28. Slot 15 passed 26px south. The next east ball is 145px north. The lip does not continue north.",
    phases: [
        #{ title: "Step", detail: "LEFT to x=4888, then up to four north steps" },
        #{ title: "Record", detail: "quote the grade and do not call the pin the door" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_22. Do not open Play_eog9_10 through Play_eog9_21. Do not replay the stair. Do not walk the 29-step line. Do not press UP before x is back near 4888. Do not walk west of x=4884. The hold at (4875, 5895) dropped HP to 24. Standing at (4904, 5886) kept HP 28 while the east ball passed. Columns 37-38 are wall on the next row north, so do not try to walk north from x=4904. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_22. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_22. Play_eog9_23 must be absent before the run.\n";
    p += "Describe Play_eog9_22 first. A hit is room 0xB9, layer 1, x within 8 of 4904, y within 8 of 5886, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. East-column means x >= 4884. West-column means x <= 4876. North means y less than link y. Ignore balls already south of link. The saved pose had slot 15 at (4888, 5912), 26px south, slot 13 at (4888, 5741), 145px north, and slot 14 at (4872, 5826), 60px north. Use the live slots.\n";
    p += "If an east-column ball already has abs(y - link_y) <= 24, do not move. If the nearest north east-column ball is less than 100px north, do not move and do not save. Otherwise write this prediction before input: LEFT until x is at most 4892 and at least 4884, then at most four north steps, x stays in that band, HP stays 28, room 0xB9, layer 1.\n";
    p += "Hold LEFT. Release when x is 4892 or less. If x drops below 4884, stop and do not save. If HP changes, stop and do not save. If 16 frames pass and x is still greater than 4892, stop and do not save.\n";
    p += "Then take north steps one at a time, at most four. Before each step, read HP and the slots. Skip the step if HP is not 28, if x is outside 4884 to 4892, or if the nearest north east-column ball is less than 64px north. A west-column ball within 40px is not a reason to skip while x stays at least 4884. Press UP until y decreases by at least 8, or 8 frames pass. If HP changes, stop and do not save. If y does not decrease by 8, stop the climb. Do not press LEFT or RIGHT during the climb.\n";
    p += "Save Play_eog9_23 only when the file is absent, at least one north step matched, x is at least 4884 and at most 4892, y is at least 8px north of 5886, HP is 28, the nearest north east-column ball is more than 40px north, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, xy when LEFT released, xy after each north step, final xy, final ball xy, final HP, and whether the file was saved).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Play_eog9_22 is (4904, 5886), layer 1, room 0xB9, HP 28, kind resumed, source_run Play_eog9_21, frame 63. It is not the door. RIGHT released at (4904, 5886). The hold was 58 frames. Slot 15 ended at (4888, 5912), 26px south. Slot 13 was at (4888, 5741), 145px north. Slot 14 was at (4872, 5826), 60px north. Play_eog9_21 remains (4897, 5886). The hold at (4875, 5895) dropped HP to 24 and was not saved. The 82-frame gap is not a route. The 29-step line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If this run saved Play_eog9_23, quote its xy and HP and say it is not the door. If the west step or the climb missed, or HP fell, say the xy and do not invent the pin.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, and Play_eog9_9 through Play_eog9_22. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if a pin that was not saved is described as a file on disk, if (4875, 5895) is called safe or saved, if Play_eog9_22 or Play_eog9_23 is called the door, if the hold at (4881, 5969) is called safe, if a pin was saved after an HP drop, if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if (4896, 5977) is called a door approach, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_22 is (4904, 5886), HP 28, and it is not the door. The hold at (4875, 5895) dropped HP to 24 and was not saved. Do not describe Play_eog9_23 as saved unless the act note says the file was written. The 82-frame gap is not a route. The 29-step line is withdrawn. No save after an HP drop. (4896, 5977) is not a door approach. No pin from this hall is the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-back-act",
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
        label: "b9-back-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-back-skeptic",
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
            label: "b9-back-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
