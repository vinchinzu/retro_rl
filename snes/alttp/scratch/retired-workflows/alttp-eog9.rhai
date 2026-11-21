let meta = #{
    name: "alttp-eog9",
    description: "Sanctuary heart, overworld, and Eastern lobby under the assisted spine",
    when_to_use: "rr-eog9.1 and rr-eog9.2 are closed. Walk rr-eog9.3 through rr-eog9.8 in order and stop when a bead does not close.",
    phases: [
        #{ title: "Chest", detail: "rr-eog9.1 already closed" },
        #{ title: "Exit", detail: "rr-eog9.2 already closed" },
        #{ title: "Overworld", detail: "rr-eog9.3 screens to 0x1E" },
        #{ title: "Hut", detail: "rr-eog9.4 Sahasrahla" },
        #{ title: "Maze", detail: "rr-eog9.5 hedge plaza" },
        #{ title: "Door", detail: "rr-eog9.6 room 0xC9" },
        #{ title: "Spine", detail: "rr-eog9.7 power-on tape" },
        #{ title: "Tests", detail: "rr-eog9.8 pytest" },
        #{ title: "Report", detail: "epic close and export" },
    ],
};

let lane_schema = #{
    "type": "object",
    "required": ["closed", "bead", "glance", "blocker"],
    "properties": #{
        "closed": #{ "type": "boolean" },
        "bead": #{ "type": "string" },
        "glance": #{ "type": "string" },
        "blocker": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "epic_closed"],
    "properties": #{
        "summary": #{ "type": "string" },
        "epic_closed": #{ "type": "boolean" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Read snes/alttp/AGENTS.md and snes/alttp/docs/tasks/residual.md with read_file before you act. Session skill: .grok/skills/alttp-session/SKILL.md. Assist contract: snes/alttp/docs/ASSIST_CONTRACT.md.\n";
    p += "This is the assisted Sanctuary-to-Eastern chain, epic rr-eog9. Beads are the tracker. Do not edit PORT_PLAN.md. Do not mark AP-105 done. AP-082 stays the Clean rung. Do not edit docs/STATUS.md. Do not move the continuous tip off room 0x50.\n";
    p += "Claim exactly one bead, the id in your task. If the epic is still open, set rr-eog9 in_progress first, then bd update <id> --status in_progress. Do not claim a second bead. On a real glance, bd close <id> --reason with the room or screen, xy, and the bytes below. On a miss, bd update <id> --append-notes with the measured pose and leave it in_progress. Do not close the epic. Do not run bd export. Do not git add, commit, or push.\n";
    p += "No RAM writes of max HP, current HP except disclosed Survival refills already in the sanctuary controller, lamp, rupees, bombs, sword, keys, big key, pendant $F374, or follower $F3CC. A chest or dialogue byte must rise source=play. A poke cannot close the bead.\n";
    p += "Do not load EasternPalaceEntry, any Whatif_ pin, or SanctuaryExit as this chain's predecessor. Do not overwrite recordings/verified_tip_run.json or recordings/sanctuary_20261001/.\n";
    p += "Geometry goes in maps/. Do not paste a coordinate list into a new Python hop. Leave proof is a RAM glance, not an MP4. Halt at the first pose that misses the predicted tile. Two tries, then stop. Do not weaken the close check.\n";
    p += "Re-read a file immediately before editing it. Soft max about 1000 lines. Merge into the existing module. No sibling extract.\n";
    p += "ROM commands use cwd snes/. Set SDL_VIDEODRIVER=dummy and SDL_AUDIODRIVER=dummy. Pytest uses QT_QPA_PLATFORM=offscreen. One emulator in your process. Do not kill other Python processes.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this bead's leftover: the glance, the state filename, and the next bead. Keep the Clean tip sentence.\n";
    return p;
}

fn stage_prompt(bead, title, body) {
    let p = rules();
    p += "You are the manager for ";
    p += title;
    p += ". Bead: ";
    p += bead;
    p += ".\n";
    p += body;
    p += "\nReturn closed true only after bd close and a non-empty glance. Otherwise closed false and put the measured miss in blocker. bead is the id you claimed.\n";
    return p;
}

fn took(r) {
    if r == () || r.success != true || r.output == () {
        return false;
    }
    if type_of(r.output) != "map" {
        return false;
    }
    if r.output.closed != true {
        return false;
    }
    if r.output.glance == () || type_of(r.output.glance) != "string" {
        return false;
    }
    let g = r.output.glance;
    g.trim();
    if g == "" {
        return false;
    }
    return true;
}

let stages = [];
stages.push(#{
    id: "rr-eog9.3",
    title: "Overworld",
    label: "mgr-ow",
    body: "rr-eog9.1 and rr-eog9.2 are already closed. Do not reopen them and do not load Play_eog9_1. Load only snes/alttp/custom_integrations/Zelda3-Snes/Play_eog9_2.state. Its provenance is recordings/sanctuary_20261001/exit_replay.json frame 37457. If that file is missing, or the live glance is not outdoors on screen 0x13 at (1880, 1083) with HP 32/32 and control ready, stop. Walk screens 0x13, 0x14, 0x15, 0x1D, then 0x1E in that order on real tiles. alttp.overworld.walk_to_screen may take each neighboring screen in turn with guard=True. A shorter grid path through 0x16 or 0x1B does not close. Cut bushes and grass beside the path. Record rupee, bomb, and heart drops that actually spawn. Zero drops after real cuts is a note, not a poke. Death, a skipped screen, or a desync does not close. Save Play_eog9_3 on screen 0x1E, outdoors, control ready. Do not load SanctuaryExit or EasternPalaceEntry.",
});
stages.push(#{
    id: "rr-eog9.4",
    title: "Hut",
    label: "mgr-hut",
    body: "Load only Play_eog9_3 on screen 0x1E. Enter Sahasrahla's hut, room 0x05. Advance the elder text with buttons so the Pendant of Courage quest is marked. $F374 staying 0 is expected. Do not claim boots. The boots receipt already measured on a later pin required the pendant. Loot pots and chests that open in play. Record rupee and bomb bytes that rise. A sealed bomb wall stays a note. Exit to screen 0x1E. Save Play_eog9_4 outdoors on 0x1E. Close only with the dialogue glance and the item bytes you measured.",
});
stages.push(#{
    id: "rr-eog9.5",
    title: "Maze",
    label: "mgr-maze",
    body: "Load only Play_eog9_4. Cross the screen 0x1E hedge from the hut to the Eastern entrance plaza. Avoid waking Armos. The plan names plaza (1984, 1144). Close only on the live elevated landing, still outdoors on 0x1E, and record the measured xy. A coordinate copied into Python is not the landing. Save Play_eog9_5 there.",
});
stages.push(#{
    id: "rr-eog9.6",
    title: "Door",
    label: "mgr-door",
    body: "Load only Play_eog9_5. Walk through eastern_palace_door_to_0xc9. Close only when the live overworld step enters room 0xC9, indoors true, game mode 0x07, control ready. Record spawn xy, sword, lamp, $F36C, and $F3CC. The plan names (4856, 6616). Loading EasternPalaceEntry, or running room_c9_north_to_0xb9 from that old pin, does not close this bead. If the live lobby can take the existing north hop with no new poke, record that after the door glance. Save Play_eog9_6 in room 0xC9.",
});
stages.push(#{
    id: "rr-eog9.7",
    title: "Spine",
    label: "mgr-spine",
    body: "This bead is a power-on tape, not a pin load. Extend the existing hop table, run_spine.py, and audit_power_on.py with --through eastern_entry so one controller covers Sanctuary, the heart chest, the south exit, screens 0x13 to 0x1E, the hut, and the room 0xC9 lobby. Write a new recording directory. Do not overwrite sanctuary_20261001 or verified_tip_run.json. Replay that tape from the first frame. Close only when the replay reports zero desyncs and zero replay state loads, and the final glance is room 0xC9 with the heart already taken in play. Survival refills stay inside the assist contract. This does not close AP-082 or AP-105. Play_eog9_6 is a dev checkpoint, not the proof.",
});
stages.push(#{
    id: "rr-eog9.8",
    title: "Tests",
    label: "mgr-tests",
    body: "Add tests for the maps and hops this chain actually landed: room_12.json chest and south door, the overworld screens that were walked, and fail-closed behavior when --through eastern_entry is missing a hop. From cwd snes/, run QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests -q. Close only when that run is green. Do not delete a failing test to make it green. Do not boot a second emulator for a test the suite already skips without a ROM.",
});

let advance = true;
let stopped = "";
let closed_ids = [];

phase("Chest");
log("rr-eog9.1 already closed");
closed_ids.push("rr-eog9.1");
phase("Exit");
log("rr-eog9.2 already closed");
closed_ids.push("rr-eog9.2");

for s in stages {
    phase(s.title);
    if advance {
        let r = agent(stage_prompt(s.id, s.title, s.body), #{
            label: s.label,
            capability_mode: "all",
            output_schema: lane_schema,
        });
        if took(r) {
            closed_ids.push(s.id);
            log(s.id + " closed");
        } else {
            advance = false;
            stopped = s.id;
            log("stopped at " + s.id);
        }
    } else {
        log("skipped " + s.id);
    }
}

phase("Report");
let report_prompt = "";
report_prompt += "Repo: /home/v/01_projects/11_games/retro_rl. Read-only except bd close on the epic and one bd export. Do not edit source, docs, or PORT_PLAN.md. Do not git add, commit, or push.\n";
report_prompt += "Run bd show rr-eog9 and bd show rr-eog9.1 through bd show rr-eog9.8. Use the live status, not this prompt.\n";
report_prompt += "The script stopped_at value is: ";
if stopped == "" {
    report_prompt += "none";
} else {
    report_prompt += stopped;
}
report_prompt += ".\n";
report_prompt += "Close epic rr-eog9 only when all eight children are closed. Then run bd export -o .beads/issues.jsonl from the repo root. If any child is open or in_progress, still export, and do not close the epic.\n";
report_prompt += "Return summary as the eight statuses plus whether the epic closed. epic_closed is true only after bd close rr-eog9 succeeded.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});

let summary = "";
let n = 0;
for id in closed_ids {
    if n > 0 {
        summary += ", ";
    }
    summary += id;
    n += 1;
}
if stopped == "" {
    summary += " closed. Chain finished.";
} else {
    summary += " closed. Stopped at ";
    summary += stopped;
    summary += ".";
}
if report != () && report.success == true && type_of(report.output) == "map" && report.output.summary != () {
    let extra = report.output.summary;
    if type_of(extra) == "string" {
        extra.trim();
        if extra != "" {
            summary += " ";
            summary += extra;
        }
    }
}
complete(#{
    summary: summary,
    stopped_at: stopped,
    closed_count: n,
});
