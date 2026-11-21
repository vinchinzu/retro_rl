let meta = #{
    name: "alttp-b9-cross",
    description: "Step west from Play_eog9_13 to x=4872 and hold until the east ball is south. Do not step north.",
    when_to_use: "After alttp-b9-dodge. Play_eog9_13 is (4888, 5969), HP 28. Slot 15 passed west. Slot 14 is on x=4888 about 60px north. Do not hold on x=4888.",
    phases: [
        #{ title: "Step", detail: "two west steps, then hold for the east ball" },
        #{ title: "Record", detail: "name Play_eog9_13 as the dodge pin and quote this grade" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_13. Do not open Play_eog9_10, Play_eog9_11, or Play_eog9_12. Do not replay the stair. Do not step north. Do not walk the 29-step line. Do not hold on x=4888. A hold at (4881, 5969) already dropped HP from 28 to 24. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_13. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_13. Play_eog9_14 must be absent before the run.\n";
    p += "Describe Play_eog9_13 first. A hit is room 0xB9, layer 1, x within 8 of 4888, y within 8 of 5969, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. A west-column ball has abs(x - 4872) <= 8. An east-column ball has abs(x - 4888) <= 8. North means y less than link y. Ignore balls already south of link. If any east-column ball already has abs(y - link_y) <= 24, do not step. If the nearest north east-column ball is at least 48px north of link y, write this prediction before input: two west steps, no pause between them, x decreases by about 16 and stays within 8 of 4872, y stays within 8 of the pin, HP stays 28, room 0xB9, layer 1.\n";
    p += "Press LEFT until x is at least 8px west of the pin or 8 frames pass. Read HP. If HP changed, stop and do not save. If x is still within 8 of 4888, stop and do not save. Then press LEFT again until x is within 8 of 4872 or 8 more frames pass. Do not press UP. If HP changes, stop and do not save.\n";
    p += "Hold still only after x is within 8 of 4872. Stop the hold when the east-column ball that was north has y greater than link_y + 24, or when a north west-column ball is within 40px of link y, or when 50 frames have passed since the open, or when HP changes. If HP changes, do not save.\n";
    p += "Save Play_eog9_14 only when the file is absent, x is within 8 of 4872, y is within 8 of 5969, HP is 28, the approaching east-column ball is south of link by more than 24px, the nearest north west-column ball is more than 48px north, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, xy after each step, frames held, final xy, final ball xy, and final HP).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "The dodge save is Play_eog9_13 at (4888, 5969), layer 1, room 0xB9, HP 28, kind resumed, source_run Play_eog9_12, frame 50. The residual sentence that calls Play_eog9_12 the saved pin of that dodge is wrong. Play_eog9_12 remains (4872, 5969). The hold at (4881, 5969) dropped HP from 28 to 24 and was not saved. The 82-frame gap is not a route. The 29-step line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If the west steps missed or HP fell, say the xy and do not call it a route. If it held, quote Play_eog9_14 and say it is not the door.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, Play_eog9_10, Play_eog9_11, Play_eog9_12, and Play_eog9_13. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if Play_eog9_12 is called the dodge save at (4888, 5969), if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if the hold at (4881, 5969) is called safe, if a pin was saved after an HP drop, if (4896, 5977) is called a door approach, if Play_eog9_13 or Play_eog9_14 is called the door, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_13 is the dodge pin at (4888, 5969), HP 28. Play_eog9_12 stays (4872, 5969). The 82-frame gap is not a route. The 29-step line is withdrawn. The hold at (4881, 5969) dropped HP to 24 and was not saved. No save after an HP drop. (4896, 5977) is not a door approach. No pin from this hall is the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-cross-act",
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
        label: "b9-cross-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-cross-skeptic",
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
            label: "b9-cross-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
