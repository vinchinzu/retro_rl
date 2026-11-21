let meta = #{
    name: "alttp-port-parallel-20",
    description: "Enter the Sanctuary secret hole from screen 0x1B without a search restore",
    when_to_use: "After alttp-port-parallel-19. The proven hole script missed at (2435,1704).",
    phases: [
        #{ title: "Hole", detail: "one Clean entry" },
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

phase("Hole");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Sanctuary Clean lane. Claim name is exactly sanctuary-clean. AP-082 is [~] sanctuary-clean. Do not claim AP-113 or AP-168.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md.\n";
prompt += "Clean power-on stopped outdoors on screen 0x1B at (2435,1704), HP 24, sword 0, follower 5, progress 0, flags 0x10. SECRET_HOLE_ENTRY_SCRIPT missed with writes=0 and no search set_state. Entering room 0x55 by the candidate search restore is forbidden. No HP refill. No RAM write. No load after power-on.\n";
prompt += "Read why that script missed. One corrected approach from that pose into room 0x55, still writes=0 and no set_state. If room 0x55 loads, continue the Clean escort only while HP stays above 0 without a refill. Mark AP-082 done only when room 0x12, $F3C5 == 2, $F3C6 == 0x15, and the follower is cleared, with zero refills and zero restores. If the hole still misses, note the pose and the attribute under Link and leave the claim [~]. Do not block AP-082.\n";
prompt += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Return completed, waiting_on, and blocked.\n";
let lane = agent(prompt, #{
    label: "mgr-sanctuary",
    capability_mode: "all",
    output_schema: lane_schema,
});
if lane == () || lane.success != true {
    log("sanctuary lane returned no success");
} else {
    log("sanctuary lane returned");
}

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP- on the same line. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the lowest open id, and the box mark for AP-082. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Twentieth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
