let meta = #{
    name: "alttp-eog9-door",
    description: "Eastern door from the hedge pin, then the power-on tape if the lobby glance closes",
    when_to_use: "After Play_eog9_5. The hedge approach is already on screen 0x1E. Do not rewalk it.",
    phases: [
        #{ title: "Door", detail: "Play_eog9_5 into room 0xC9" },
        #{ title: "Spine", detail: "power-on tape through the lobby" },
        #{ title: "Tests", detail: "the hops this chain landed" },
        #{ title: "Report", detail: "glance and tracker sentences" },
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
    "required": ["summary", "door_closed"],
    "properties": #{
        "summary": #{ "type": "string" },
        "door_closed": #{ "type": "boolean" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. There is no retro_rl checkout. Read snes/alttp/AGENTS.md with read_file before you act.\n";
    p += "This is the assisted Sanctuary-to-Eastern chain. rr-eog9.1 through the hedge approach are already recorded. Do not rewalk screens 0x13, 0x14, 0x15, 0x1D, the 0x1D north lip, the hut, or the hedge. Do not load Play_eog9_1, Play_eog9_2, Play_eog9_3, or Play_eog9_4 as this chain's start.\n";
    p += "Do not mark AP-105 done. Do not change any checkbox in snes/alttp/docs/PORT_PLAN.md. You may replace only the AP-105e, AP-105f, AP-105g, and AP-105h sentences inside that open item, and only with a glance you measured. Do not edit docs/STATUS.md. Do not run bd. There is no beads database, and you must not run bd init. Do not git add, commit, or push.\n";
    p += "No RAM writes of max HP, current HP, lamp, rupees, bombs, sword, keys, big key, pendant, or follower $F3CC. A poke cannot close the bead. Do not load EasternPalaceEntry, any Whatif_ pin, or SanctuaryExit. Do not overwrite recordings/verified_tip_run.json, recordings/clean_escape_20261002/, recordings/sanctuary_20261001/, recordings/bush_cell_20261002/, recordings/eog9_crossing_20261001/, recordings/eog9_hut_20261001/, or recordings/eog9_hedge_20261001/.\n";
    p += "Geometry stays in maps/. Do not paste a new coordinate list into Python. The Eastern door is already mapped. Leave proof is a RAM glance, not an MP4. Two tries, then stop. Do not weaken the close check.\n";
    p += "Re-read a file immediately before editing it. Soft max about 1000 lines. segment.py is 943 lines and overworld.py is 985. Do not add a new module. Merge a few lines into the existing eastern approach function, or stop if that would pass 1000. Do not edit overworld.py unless a one-line bugfix is required.\n";
    p += "ROM commands use cwd snes/. Set SDL_VIDEODRIVER=dummy and SDL_AUDIODRIVER=dummy. Pytest uses QT_QPA_PLATFORM=offscreen. One emulator in your process. Do not kill other Python processes.\n";
    p += "Overwrite snes/alttp/docs/tasks/residual.md with this bead's leftover: the glance, the state filename, and what is still open. Keep one sentence that the Clean Sanctuary tape is recordings/clean_escape_20261002/.\n";
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
    p += "\nReturn closed true only with a non-empty glance that meets the close check. Otherwise closed false and put the measured miss in blocker.\n";
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
    id: "rr-eog9.6",
    title: "Door",
    label: "mgr-door",
    body: "Load only snes/alttp/custom_integrations/Zelda3-Snes/Play_eog9_5.state. Read maps/screen_1e_eastern.json first. The hedge replay recordings/eog9_hedge_20261001/replay.json left outdoors, loaded screen 0x1E, (3920, 1582), control ready, HP 28, sword 1, lamp 1, follower 0. Standing quadrant 0x1F is not a screen change. If the live pin is not that outdoors glance, stop and do not walk. Hold UP on the mapped door eastern_door_to_0xc9. Close only when that step enters room 0xC9, indoors, module 0x07, submodule 0x00, control ready. Record spawn xy, sword, lamp, HP, max HP, follower, and keys. The landing (4856, 6616) was measured from EasternPalaceEntry and does not close this bead. Save Play_eog9_6 only after the live glance. Replay the door buttons from Play_eog9_5 into a new directory recordings/eog9_door_20261023/. Close only if that replay has zero desync, zero state loads, and zero RAM writes. A controller search restore does not count. Update the AP-105f sentence with the glance or the miss. Do not tick AP-105.",
});
stages.push(#{
    id: "rr-eog9.7",
    title: "Spine",
    label: "mgr-spine",
    body: "The door bead already closed. Read spine/hops.py and scripts/audit_power_on.py before editing. Add one power-on stop, eastern_entry, after the existing sanctuary chain. Do not remove kakariko. The controller covers Sanctuary, the heart chest, the south exit, the proved screens, the hut, the hedge, and the room 0xC9 lobby. Write recordings/eog9_spine_20261023/. Replay that tape from the first frame with blank SRAM. Close only when the replay reports zero desync and zero state loads, and the final glance is room 0xC9 with the heart already taken in play. Name any Survival refills. Do not add a new refill. Play_eog9_6 is a dev checkpoint, not the proof. This does not close AP-105. Update the AP-105g sentence with the replay glance or the miss.",
});
stages.push(#{
    id: "rr-eog9.8",
    title: "Tests",
    label: "mgr-tests",
    body: "Add tests only for the door hop and the eastern_entry stop this chain landed. From cwd snes/, run QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests/test_opening_spine.py alttp/tests/test_overworld.py -q. If those paths do not cover the new hop, run the test file you edited. Close only when that run is green. Do not delete a failing test to make it green. Do not boot a second emulator for a test the suite already skips without a ROM. Update the AP-105h sentence with the pytest count.",
});

let advance = true;
let stopped = "";
let closed_ids = [];

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
report_prompt += "Repo: /home/v/01_projects/11_games/retro. Read-only. Do not edit files. Do not run bd. Do not git add, commit, or push.\n";
report_prompt += "Read snes/alttp/docs/tasks/residual.md and the AP-105 breakdown in snes/alttp/docs/PORT_PLAN.md. Confirm whether Play_eog9_6.state exists. The script stopped_at value is: ";
if stopped == "" {
    report_prompt += "none";
} else {
    report_prompt += stopped;
}
report_prompt += ".\n";
report_prompt += "door_closed is true only if residual.md quotes room 0xC9 indoors with control ready from Play_eog9_5, not from EasternPalaceEntry. summary: that glance or the miss, whether the power-on tape exists, and the lowest still-open child of AP-105. Do not claim the bot plays to credits.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "read-only",
    output_schema: report_schema,
});

let summary = "Door chain";
if stopped == "" {
    summary += " finished.";
} else {
    summary += " stopped at ";
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
    closed_count: closed_ids.len(),
});
