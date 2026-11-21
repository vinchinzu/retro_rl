let meta = #{
    name: "alttp-port-next",
    description: "Second ALTTP port wave: ready tasks plus the courtyard-door red",
    when_to_use: "Continue PORT_PLAN after alttp-port. Ready tasks only, plus the pocket-door tip.",
    phases: [
        #{ title: "Ready lanes", detail: "eight file-disjoint managers" },
        #{ title: "Report", detail: "tick counts" },
    ],
};

let lane_schema = #{
    "type": "object",
    "required": ["completed", "waiting_on", "blocked"],
    "properties": #{
        "completed": #{ "type": "array", "items": #{ "type": "string" } },
        "waiting_on": #{ "type": "string" },
        "blocked": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "open_count", "done_count", "blocked_count"],
    "properties": #{
        "summary": #{ "type": "string" },
        "open_count": #{ "type": "integer" },
        "done_count": #{ "type": "integer" },
        "blocked_count": #{ "type": "integer" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md only. Do not run bd. Do not git add or commit.\n";
    p += "Claim and finish only with python3 /home/v/.grok/alttp_port_claim.py (claim|done|block|note). A blocked task may be claimed again.\n";
    p += "Read the rules at the top of PORT_PLAN.md once, then only your tasks. Touch only the files those tasks list, plus a fix file named below.\n";
    p += "Two honest tries, then block. Do not weaken a Done line. Leave the AP-002 Owner ruling blank.\n";
    p += "cwd for ROM tools is snes/. SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Do not overwrite recordings/verified_tip_run.json on a red run. No RAM writes on a natural or continuous run. z3-json-data beats hint lines.\n";
    p += "Re-read a file immediately before editing it. Do not remove another task's functions.\n";
    return p;
}

fn lane_prompt(agent, tasks, extra) {
    let p = rules();
    p += "You are manager " + agent + ". Tasks, in order: " + tasks + ".\n";
    p += extra;
    p += "\nSkip a task that is already [x]. If a Needs id is not [x] and is not a task you just finished, stop and set waiting_on. Do not mark that task blocked.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Ready lanes");
let jobs = [];
jobs.push(#{
    prompt: lane_prompt(
        "mgr-tip",
        "AP-018, AP-023",
        "The fresh opening tip dies at pocket_to_main_hall approach (2048,1955) around frame 11071. recordings/verified_tip_run.json is an older green at 11245 frames; leave it on a red. The tape recordings/verified_tip.tape.npz still replays with desynced 0. AP-050 turned the room_step walker on for door exits. Check that hook first. You may edit the opening-route walker, room_engine.py, and the courtyard map. Do not edit ledger.py, rollout.py, primitives.py, ram.py, or overworld.py. If run_to_verified_tip.py is green with writes 0, mark AP-018 done. If run_spine.py --through room_50 --no-video is green and --through zelda fails closed, mark AP-023 done. If the door is still red after two tries, leave both blocked and add a note."
    ),
    label: "mgr-tip",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-ow",
        "AP-053, AP-084, AP-086",
        "Own overworld.py and the screen maps AP-086 names. Do not edit room_engine.py, nav.py, or rollout.py. Stop before AP-085."
    ),
    label: "mgr-ow",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-roll",
        "AP-056, AP-062, AP-064",
        "Own rollout.py. AP-062 is blocked: a safe first hit on CastleB1Guard is followed by a body hit. Hitstun is bit 7 of $0EF0 and strike_if_safe will not roll while it is set. Wait that bit out, then swing again. The Done line stays 10/12 offsets, 0 damage, soldier dead. AP-064 is blocked because castle soldier pins spawn no arrow. Find an archer room from z3-json-data and measure the shield there. Do not edit nav.py, room_engine.py, or primitives.py."
    ),
    label: "mgr-roll",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-prim",
        "AP-065, AP-067",
        "Own primitives.py. Stop before AP-066 and set waiting_on to AP-041. Do not poke a bottle pin. Do not edit rollout.py or room_engine.py."
    ),
    label: "mgr-prim",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-ledger",
        "AP-043, AP-070, AP-069",
        "Own ledger.py. For AP-043, census recordings/verified_tip.tape.npz and recordings/scan/index.json. Do not boot a fresh opening tip. For AP-069 and AP-070 use a castle pin, not the opening tip. Do not edit room_engine.py."
    ),
    label: "mgr-ledger",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-stairs",
        "AP-052",
        "Put transition_kind in room_timer.py only. Do not edit nav.py or room_engine.py."
    ),
    label: "mgr-stairs",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-items",
        "AP-087, AP-088",
        "Own alttp/docs/routes/ITEM_ROUTE.md. Prices in AP-088 must be measured in game. z3 logic is the authority. The Tompa movie is only a hint."
    ),
    label: "mgr-items",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lane_prompt(
        "mgr-owflags",
        "AP-010",
        "AP-010 is blocked. $F280+screen stayed 0 across the bush lift at (2431,1704) into room 0x55, and $F280-$F2FF stayed 0. Save diffs seen were sword, shield, $F3C5, $F3C6, $F3C8. Those are not the hole bit. Find the overworld overlay bit from the disassembly and a new diff. Do not repeat the $F280-only experiment. Own ram.py, its test, and one ram_map.md section."
    ),
    label: "mgr-owflags",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(jobs);

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python.\n";
report_prompt += "A heading matches ^- [.] and contains **AP-NNN. open_count is a space box. done_count is x. blocked_count is !.\n";
report_prompt += "summary: those three numbers, the blocked ids, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Ready lanes finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
