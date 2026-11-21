let meta = #{
    name: "alttp-port-eastern",
    description: "Eastern Palace steps 1-6, the census, and the overworld legs that are ready",
    when_to_use: "Continue PORT_PLAN after AP-035 is done and power-on is in room 0x81.",
    phases: [
        #{ title: "Side lanes", detail: "census, doorways, overworld, cell retry" },
        #{ title: "Eastern", detail: "steps 1 through 6" },
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
    p += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py (claim|done|block|note).\n";
    p += "Read the dungeon task template and your own section. Do not read other dungeons. Hints lose to z3-json-data and the live scan.\n";
    p += "Two honest tries, then block. Do not weaken a Done line. Leave the Owner ruling blank.\n";
    p += "ROM commands: cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "A pin is not continuous. No RAM writes on a natural or continuous run. Do not overwrite recordings/verified_tip_run.json on a red run. Never poke $F3CC.\n";
    p += "Re-read a file immediately before editing it. Add hop rows; do not rewrite the whole hop table.\n";
    return p;
}

fn lane_prompt(agent, tasks, extra) {
    let p = rules();
    p += "You are manager " + agent + ". Tasks, in order: " + tasks + ".\n";
    p += extra;
    p += "\nSkip a task that is already [x]. If a Needs id is not [x], stop and set waiting_on, unless the prompt tells you to skip that id and continue.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Side lanes");
let first = [];
first.push(#{
    prompt: lane_prompt(
        "mgr-census",
        "AP-039",
        "Own room_census.py and the generated ROOM_CENSUS.md. Do not edit hops.py or room_engine.py."
    ),
    label: "mgr-census",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-doorway",
        "AP-051",
        "Own the doorway alignment in nav.py or room_engine.py, whichever the task names. One castle door, no hand-copied approach point. Do not edit hops.py."
    ),
    label: "mgr-doorway",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-ow",
        "AP-086, AP-085, AP-093, AP-094, AP-096",
        "Skip AP-089, AP-090, AP-091, and AP-092. Their natural legs need a pendant, the push-block, or the Master Sword, which are not done. Do not stop the lane on those. AP-086 writes screen maps and one isolated walk-in each. AP-085 is a 5-screen walk under PolicyGuard. AP-093 adds swim constants. AP-094 adds flute_to. AP-096 buys or catches one bottle in play. SanctuaryExit.state is a resumed pin on screen 0x13, not a continuous claim."
    ),
    label: "mgr-ow",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-cell",
        "AP-076",
        "Start from Full_room_81, the power-on arrival at about (996,4485) in room 0x81. The previous try opened object 0x18 from inside room 0x80 (UP at x=352) and stayed in 0x80; Zelda text is y=4168. Do not repeat that. Isolate the live 0x81 to 0x80 transition and set landingXy. If two tries never change rooms, leave AP-076 blocked."
    ),
    label: "mgr-cell",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(first);

phase("Eastern");
let eastern = agent(lane_prompt(
    "mgr-eastern",
    "AP-099, AP-100, AP-101, AP-102, AP-103, AP-104",
    "Eastern Palace only. Pin EasternPalaceEntry is room 0xC9. The first forward edge is north, not south_to_overworld. Read the eight-step template. Step 3 needs AP-051; if that tick is not [x], stop there. Boss step 6 needs at least 10 of 12 offsets. Stop before AP-105. Stalfos bones are a note for the arrow task, not a reason to retick AP-064."
), #{ label: "mgr-eastern", capability_mode: "all", output_schema: lane_schema });
if eastern == () || eastern.success != true {
    log("eastern manager returned no success");
}

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python.\n";
report_prompt += "A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !.\n";
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
    summary: "Eastern wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
