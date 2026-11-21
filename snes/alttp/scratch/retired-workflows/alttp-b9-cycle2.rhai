let meta = #{
    name: "alttp-b9-cycle2",
    description: "One north step from Play_eog9_15, then dodge west and let the east ball pass.",
    when_to_use: "After alttp-b9-cycle. Play_eog9_15 is (4888, 5961), HP 28. Slot 13 is 26px south. Slot 12 is the east ball at (4888, 5901). Do not stand for it.",
    phases: [
        #{ title: "Step", detail: "one north step, then LEFT to x=4876, then hold for the east ball" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_15. Do not open Play_eog9_10 through Play_eog9_14. Do not replay the stair. Do not walk the 29-step line. Do not take a second north step. Do not stop LEFT at x=4880. Do not walk west of x=4872. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_15. frame is the frame count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_15. Play_eog9_16 must be absent before the run.\n";
    p += "Describe Play_eog9_15 first. A hit is room 0xB9, layer 1, x within 8 of 4888, y within 8 of 5961, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop.\n";
    p += "Read the cannonball slots before input. A west-column ball has abs(x - 4872) <= 8. An east-column ball has abs(x - 4888) <= 8. North means y less than link y. Ignore balls already south of link. The saved pose had slot 12 at (4888, 5901), 60px north, and slot 13 at (4872, 5987), 26px south. Use the live slots.\n";
    p += "If any east-column ball already has abs(y - link_y) <= 24, do not move. If the nearest north east-column ball is less than 56px north, do not move and do not save. If it is at least 56px north, write this prediction before input: one north step of at most 8px, then LEFT until x is 4876 or less and at least 4868, y stays within 8 of the post-step y, HP stays 28, room 0xB9, layer 1.\n";
    p += "Press UP until y has decreased by at least 8, or 8 frames pass. Read HP. If HP changed, or y decreased by less than 8, stop and do not save. Do not press UP again.\n";
    p += "Then hold LEFT. Read xy and HP every frame. Release LEFT when x is 4876 or less. If x drops below 4868, stop and do not save. If HP changes, stop and do not save. If 16 frames of LEFT pass and x is still greater than 4876, stop and do not save. x=4880 is the midpoint. Do not release there.\n";
    p += "Hold still after the release. Stop that hold when the east-column ball that was north has y greater than link_y + 24, or when a north west-column ball is within 40px of link y, or when 80 frames have passed since the open, or when HP changes. If HP changes, do not save.\n";
    p += "Save Play_eog9_16 only when the file is absent, x is 4876 or less and at least 4868, y is at least 8px north of 5961, HP is 28, the approaching east-column ball is south of link by more than 24px, the nearest north west-column ball is more than 48px north, room is 0xB9, and layer is 1.\n";
    p += "Return stepped, saved, hp, and note (pin glance, ball xy before input, prediction, xy after the north step, xy when LEFT released, frames held, final xy, final ball xy, final HP, and whether the file was saved).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Play_eog9_15 is (4888, 5961), layer 1, room 0xB9, HP 28, kind resumed, source_run Play_eog9_14, frame 52. It is not the door. The north step landed (4876, 5961). RIGHT released at (4888, 5961). The hold was 39 frames. Slot 13 ended at (4872, 5987), 26px south. Slot 12 was at (4888, 5901), 60px north. Play_eog9_14 remains (4876, 5969). The hold at (4881, 5969) dropped HP from 28 to 24 and was not saved. The 82-frame gap is not a route. The 29-step line stays withdrawn. (4896, 5977) does not extend north. Byte 0x1E is unknown, not a stair.\n";
    p += "Quote the act note. If this run saved Play_eog9_16, quote its xy and HP and say it is not the door. If it did not save, or HP fell, say the xy and do not invent the pin.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9 through Play_eog9_15. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp.\n";
    p += "Count a problem if a pin that was not saved is described as a file on disk, if Play_eog9_15 or Play_eog9_16 is called the door, if the hold at (4881, 5969) is called safe, if a pin was saved after an HP drop, if the 82-frame gap is called a route, if the 29-step line is offered as the walk, if (4896, 5977) is called a door approach, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all four files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. Play_eog9_15 is (4888, 5961), HP 28, and it is not the door. Do not describe Play_eog9_16 as saved unless the act note says the file was written. The 82-frame gap is not a route. The 29-step line is withdrawn. The hold at (4881, 5969) dropped HP to 24 and was not saved. No save after an HP drop. (4896, 5977) is not a door approach. No pin from this hall is the door. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Step");
let acted = agent(act_prompt(), #{
    label: "b9-cycle2-act",
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
        label: "b9-cycle2-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-cycle2-skeptic",
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
            label: "b9-cycle2-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    stepped: stepped,
    note: record_note,
});
