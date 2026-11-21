let meta = #{
    name: "alttp-b9-gap",
    description: "Time the Eastern cannonballs from Play_eog9_10, then take at most one north step.",
    when_to_use: "After alttp-b9-clear. Unexposed north from (4896, 5977) is 0 steps. Do not walk the 29-step line.",
    phases: [
        #{ title: "Time", detail: "stand still, log the balls, one step only if a gap exists" },
        #{ title: "Record", detail: "quote the gap and the grade" },
    ],
};

let act_schema = #{
    "type": "object",
    "required": ["gap_frames", "stepped", "saved", "hp", "note"],
    "properties": #{
        "gap_frames": #{ "type": "integer" },
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
    p += "Repo: /home/v/01_projects/11_games/retro. One emulator. Open Play_eog9_10. Do not open Play_eog9_11 for the step. Do not replay the stair. Do not walk the 29-step line. Do not walk east. Read pin_probe.open_pin and describe, and ram.read_snapshot, before any input. Read write_state_bytes, state_path, and write_pin_provenance before any save. Kind is resumed. Kind play is rejected. source_run is Play_eog9_10. frame is the step count.\n";
    p += "PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes and SDL_VIDEODRIVER=dummy from the repo root. Script in /tmp, then delete it.\n";
    p += "Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit. Do not press A. Do not refill. Do not swing the sword. Do not overwrite Play_eog9_1 through Play_eog9_11.\n";
    p += "Describe Play_eog9_10 first. A hit is room 0xB9, layer 1, x within 8 of 4872, y within 8 of 5977, HP 28/32, sword 1, lamp 1, keys 0, follower 0. If that misses, stop and do not log a fake gap.\n";
    p += "Hold still for 240 frames. Every frame, record cannonball slots: id, x, y, state, hp. A same-column ball is one with abs(x - 4872) <= 8. A frame is clear when no same-column ball has abs(y - link_y) <= 24. gap_frames is the longest run of clear frames. Write snes/alttp/docs/routes/full_clear/b9_balls.json with frames, gap_frames, and the per-slot min and max y. This file is the only game file you write besides a pin.\n";
    p += "If gap_frames is less than 16, do not move. If it is at least 16, wait until the next clear run begins, then take one north step of at most 8px. Write the predicted x, y, and HP 28 before that step. Grade after it. If HP changes, stop. Do not take a second step.\n";
    p += "Save Play_eog9_12 only when the file is absent, the one step matched within 8px, HP is 28, room is 0xB9, and layer is 1.\n";
    p += "Return gap_frames, stepped (true only if that one step was taken), saved (true only if Play_eog9_12.state was created), hp after the run, and note (the pin glance, the clear-run length, xy, sword, lamp, keys, follower, prediction, and grade).\n";
    return p;
}

fn record_prompt(act_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Act note: ";
    p += act_note;
    p += "\n";
    p += "Read b9_balls.json, b9_clear.json, b9_clear_check.json, residual.md, maps/room_b9.json, ORDER.md, and PLAY.md under snes/alttp. Use read_file.\n";
    p += "Both clear files have north_steps 0, end_x -1, and end_y -1. The unexposed floor at (4896, 5977) does not extend north. That is why no walk ran. Play_eog9_11 still holds that pose and is not a door approach. The 29-step x=4872 line stays withdrawn. Slot 15 met Link at (4872, 5947). Byte 0x1E is unknown, not a stair.\n";
    p += "Quote gap_frames from b9_balls.json and the act note. If the step was not taken, say so. If HP fell, say the xy and do not call it a route.\n";
    p += "Do not mark an ORDER row after 4 true. Do not add approachXy or a north_ point. Do not press A. Do not claim a credits tape or finished autoplay. Do not tick AP-105. Do not edit PORT_PLAN.md, docs/STATUS.md, room_engine.py, or primitives.py. Do not run bd. Do not git commit.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this sitting. Keep the Play_eog9_7 glance, the stair landing, Play_eog9_9, Play_eog9_10, and Play_eog9_11. Edit the same claims in the map note, the ORDER frontier paragraph, and PLAY.md. Return path residual.md and note.\n";
    return p;
}

fn doc_skeptic_prompt() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Read residual.md, maps/room_b9.json, ORDER.md, PLAY.md, and b9_balls.json under snes/alttp.\n";
    p += "Count a problem if the 29-step line is offered as the walk, if more than one north step is claimed, if a pin was saved after an HP drop, if (4896, 5977) is called a door approach, if gap_frames from b9_balls.json is not quoted, if 0x1E is called a stair, if approachXy is set, if a north_ point is added, if an ORDER row after 4 is true, if a credits tape or finished autoplay is claimed, if the text says to press A, or if AP-105 is marked done. Quote each line.\n";
    p += "ok is true only when all five files were read and none of those are present. Do not edit. Do not boot.\n";
    p += "Return ok, problems, and note.\n";
    return p;
}

fn fix_prompt(problems) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. Skeptic note: ";
    p += problems;
    p += "\n";
    p += "Edit only residual.md, the notes in maps/room_b9.json, the ORDER frontier paragraph, and the stale sentences in PLAY.md. The 29-step line is withdrawn. One north step at most. No save after an HP drop. (4896, 5977) is not a door approach. 0x1E is unknown, not a stair. Rows 1-4 are the only true rows. No approachXy, no north_ point, no press A, no credits tape, AP-105 stays open, autoplay is not finished. Quote gap_frames from b9_balls.json. Do not edit other files. Do not boot. Do not run bd. Do not git commit.\n";
    p += "Return path residual.md and note.\n";
    return p;
}

phase("Time");
let acted = agent(act_prompt(), #{
    label: "b9-gap-act",
    capability_mode: "all",
    output_schema: act_schema,
});
let act_note = "no act";
let gap = -1;
if acted != () && acted.success && type_of(acted.output) == "map" {
    if acted.output.note != () { act_note = acted.output.note; }
    if acted.output.gap_frames != () { gap = acted.output.gap_frames; }
}
log(act_note);

phase("Record");
let record_note = "no record";
if gap >= 0 {
    let recorded = agent(record_prompt(act_note), #{
        label: "b9-gap-record",
        capability_mode: "read-write",
        output_schema: record_schema,
    });
    if recorded != () && recorded.success && type_of(recorded.output) == "map" && recorded.output.note != () {
        record_note = recorded.output.note;
    }
    let skeptic = agent(doc_skeptic_prompt(), #{
        label: "b9-gap-skeptic",
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
            label: "b9-gap-fix",
            capability_mode: "read-write",
            output_schema: record_schema,
        });
    }
}

complete(#{
    gap_frames: gap,
    note: record_note,
});
