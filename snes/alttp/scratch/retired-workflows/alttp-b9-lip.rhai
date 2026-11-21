let meta = #{
    name: "alttp-b9-lip",
    description: "One north step from Play_eog9_20, then step east to x=4896. Do not press LEFT.",
    when_to_use: "After alttp-b9-mid. Standing at (4875, 5895) dropped HP to 24 while slot 11 was 10px north. Passing x=4880 kept HP 28. Do not wait beside that ball.",
    phases: [
        #{ title: "Step", detail: "one north step, then RIGHT to the east lip" },
        #{ title: "Record", detail: "quote the grade and do not call a hit a route" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_20. Do not open Play_eog9_10 through Play_eog9_19. Do not replay the stair. Do not walk the 29-step line. Do not press LEFT. The hold at (4875, 5895) dropped HP from 28 to 24 while slot 11 was at (4880, 5885), 10px north. That pose was not saved. Passing x=4880 at frame 5 kept HP 28. Do not stand on x=4876. Do not take a second north step. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_20. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_20. Play_eog9_21 must be absent before the run.\n";
    p += "Describe Play_eog9_20 first. A hit is room 0xB9, layer 1, x within 8 of 4888, y within 8 of 5894, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. Mid means x from 4877 to 4883. East-column means x >= 4884. North means y less than link y. Ignore balls already south of link. The saved pose had slot 11 at (4880, 5842), 52px north, and slot 15 at (4888, 5748), 146px north. Use the live slots.\n";
    p += "If a mid ball already has abs(y - link_y) <= 24, do not move. If the nearest north mid ball is less than 48px north, do not move and do not save. Otherwise write this prediction before input: one north step, then RIGHT until x is at least 4896 and at most 4904, HP stays 28, room 0xB9, layer 1.\n";
    p += "Press UP until y has decreased by at least 8, or 8 frames pass. If HP changed, or y decreased by less than 8, stop and do not save. Do not press UP again.\n";
    p += "Hold RIGHT. Release when x is at least 4896. If x passes 4904, stop and do not save. If HP changes, stop and do not save. If 12 frames of RIGHT pass and x is still below 4896, stop and do not save. That means the east lip was not entered.\n";
    p += "Hold still after the release. Stop when the mid ball that was north has y greater than link_y + 24, or a north east-column ball is within 24px, or HP changes, or 70 frames have passed since the open. If HP changes, do not save.\n";
    p += "Save Play_eog9_21 only when the file is absent, x is within 8 of 4896, y is at least 8px north of 5894, HP is 28, that mid ball is south of link by more than 24px, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, xy after the north step, xy when RIGHT released, frames held, final xy, final ball xy, final HP, and whether the file was saved).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Play_eog9_20 is (4888, 5894), layer 1, room 0xB9, HP 28, kind resumed, source_run Play_eog9_19, frame 52. It is not the door. The mid dodge reached (4876, 5894) at frame 8 with HP 28, then a 21-frame hold ended at (4875, 5895) with HP 24. Slot 11 was at (4880, 5885), 10px north. Play_eog9_21 was not saved. Play_eog9_19 remains (4876, 5902). (4876, 5937) is not a saved pin. The hold at (4881, 5969) dropped HP from 28 to 24 and was not saved. The 82-frame gap is not a route. The 29-step line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If this run saved Play_eog9_21, quote its xy and HP and say it is not the door. If RIGHT never reached x=4896, or HP fell, say the xy and do not invent the pin.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, and Play_eog9_9 through Play_eog9_20. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if a pin that was not saved is described as a file on disk, if (4875, 5895) is called safe or saved, if (4876, 5937) is called a saved pin, if Play_eog9_20 or Play_eog9_21 is called the door, if the hold at (4881, 5969) is called safe, if a pin was saved after an HP drop, if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if (4896, 5977) is called a door approach, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_20 is (4888, 5894), HP 28, and it is not the door. The hold at (4875, 5895) dropped HP to 24 and was not saved. (4876, 5937) is not a saved pin. Do not describe Play_eog9_21 as saved unless the act note says the file was written. The 82-frame gap is not a route. The 29-step line is withdrawn. The hold at (4881, 5969) dropped HP to 24 and was not saved. No save after an HP drop. (4896, 5977) is not a door approach. No pin from this hall is the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-lip-act",
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
        label: "b9-lip-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-lip-skeptic",
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
            label: "b9-lip-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
