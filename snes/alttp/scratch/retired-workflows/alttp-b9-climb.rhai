let meta = #{
    name: "alttp-b9-climb",
    description: "From Play_eog9_17, let the west ball pass, then take up to four north steps on x=4888.",
    when_to_use: "After alttp-b9-pair. Play_eog9_17 is (4888, 5945), HP 28. The near ball is west, 60px north. Half B stepped into that ball and did not save.",
    phases: [
        #{ title: "Step", detail: "hold for the west ball, then up to four north steps" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_17. Do not open Play_eog9_10 through Play_eog9_16. Do not replay the stair. Do not walk the 29-step line. Do not press LEFT. Do not press RIGHT. Do not step west. The pair's Half B reached (4876, 5937), HP 28, and saved nothing because a west ball was 33px north. That gate was wrong for this pin. A west ball within 40px is not a reason to stop while x stays on 4888. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_17. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_17. Play_eog9_18 must be absent before the run.\n";
    p += "Describe Play_eog9_17 first. A hit is room 0xB9, layer 1, x within 8 of 4888, y within 8 of 5945, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. A west-column ball has abs(x - 4872) <= 8. An east-column ball has abs(x - 4888) <= 8. North means y less than link y. Ignore balls already south of link. The saved pose had slot 15 at (4872, 5885), about 60px north, and slot 13 at (4888, 5714), about 231px north. Use the live slots.\n";
    p += "If an east-column ball already has abs(y - link_y) <= 24, do not move. If the nearest north west-column ball is less than 48px north, do not move. Otherwise write this prediction before input: hold still until that nearest west-column ball is more than 24px south, then take at most four north steps of 8px, x stays within 8 of 4888, HP stays 28, room 0xB9, layer 1.\n";
    p += "Hold still with no direction held. Stop the hold when that west-column ball has y greater than link_y + 24, or an east-column ball is within 48px, or HP changes, or 80 frames pass. If HP changes, stop and do not save. If 80 frames pass before the west ball is 24px south, stop and do not save.\n";
    p += "Then take north steps one at a time, at most four. Before each step, read HP and the slots. Skip the step if HP is not 28, if x is not within 8 of 4888, if an east-column ball is within 56px north, or if four steps are done. Press UP until y decreases by at least 8, or 8 frames pass. If HP changes, stop the climb and do not save. If y does not decrease by 8, stop the climb. Do not press LEFT or RIGHT.\n";
    p += "Save Play_eog9_18 only when the file is absent, at least one north step matched, x is within 8 of 4888, y is at least 8px north of 5945, HP is 28, the nearest north east-column ball is more than 48px north, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, xy when the west ball cleared, xy after each north step, final xy, final ball xy, final HP, and whether the file was saved).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Play_eog9_17 is (4888, 5945), layer 1, room 0xB9, HP 28, kind resumed, source_run Play_eog9_16, frame 52. It is not the door. Play_eog9_16 remains (4876, 5953). Half B of the pair reached (4876, 5937), HP 28, held 0 frames, and did not save. That xy is not a pin. The hold at (4881, 5969) dropped HP from 28 to 24 and was not saved. The 82-frame gap is not a route. The 29-step line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If this run saved Play_eog9_18, quote its xy and HP and say it is not the door. If it did not save, or HP fell, say the xy and do not invent the pin.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, and Play_eog9_9 through Play_eog9_17. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if a pin that was not saved is described as a file on disk, if (4876, 5937) is called a saved pin, if Play_eog9_17 or Play_eog9_18 is called the door, if the hold at (4881, 5969) is called safe, if a pin was saved after an HP drop, if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if (4896, 5977) is called a door approach, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_17 is (4888, 5945), HP 28, and it is not the door. (4876, 5937) is not a saved pin. Do not describe Play_eog9_18 as saved unless the act note says the file was written. The 82-frame gap is not a route. The 29-step line is withdrawn. The hold at (4881, 5969) dropped HP to 24 and was not saved. No save after an HP drop. (4896, 5977) is not a door approach. No pin from this hall is the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-climb-act",
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
        label: "b9-climb-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-climb-skeptic",
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
            label: "b9-climb-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
