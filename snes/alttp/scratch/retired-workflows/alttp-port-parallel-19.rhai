let meta = #{
    name: "alttp-port-parallel-19",
    description: "Compose a Clean power-on to Sanctuary with no HP refill and no search restore",
    when_to_use: "After the Sanctuary audit closed AP-076 through AP-081. AP-082 stays open on seven refills and seven restores.",
    phases: [
        #{ title: "Sanctuary", detail: "Clean rung C1" },
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

phase("Sanctuary");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Sanctuary Clean lane. Claim AP-082. Do not claim AP-113 or AP-168.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md, kholdstare.py, or moldorm.py.\n";
prompt += "AP-076 through AP-081 are done. The Survival tape is recordings/sanctuary_20261001/. The measured cell route is 0x81 to 0x71 to 0x70 to 0x80, with no west exit. The escort rooms are 0x80, 0x70, 0x71, 0x72, 0x02, 0x11, 0x12. Follower $F3CC reaches 1 in the cell from the real predecessor, and it is cleared at Sanctuary. Do not put the west-edge cell hypothesis back.\n";
prompt += "AP-082 closes only on a Clean composition: no load, no RAM write, no HP refill, and no search restore. The predicate is room 0x12, progress $F3C5 == 2, flags $F3C6 == 0x15, follower cleared. Read tests/test_escort_to_sanctuary.py and story_glance before you boot.\n";
prompt += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes.\n";
prompt += "If the run would need an HP refill or a search restore, stop at that room, note the HP and the room, and leave AP-082 open. Do not block it. Do not mark it done from the Survival tape.\n";
prompt += "Return completed, waiting_on, and blocked.\n";
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP- on the same line. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the lowest open id, and the box mark for AP-082, AP-113, and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Nineteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
