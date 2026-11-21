let meta = #{
    name: "alttp-port-parallel-22",
    description: "Cross room 0x80 off the two flail hits that killed the Clean escort",
    when_to_use: "After alttp-port-parallel-21. The bush entry works. The flail kills at (405,4286) and (410,4285).",
    phases: [
        #{ title: "Flail", detail: "one path off those hits" },
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

phase("Flail");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Sanctuary Clean lane. Claim name is exactly sanctuary-clean. AP-082 is [~] sanctuary-clean. Do not claim AP-113 or AP-168.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md.\n";
prompt += "Power on. No load, no RAM write, no HP refill, no search restore. Enter room 0x55 from screen 0x1B pose (2431,1704) with SECRET_HOLE_ENTRY_SCRIPT. That entry is already proven at (2975,2710) HP 24.\n";
prompt += "Two room 0x80 deaths are closed. (303,4232) HP 0. And the south-east corner (393,4280) at HP 20, then flail hits (405,4286) for 20 to 12 and (410,4285) for 12 to 4, death (424,4287) HP 0 mode 0x12. Follower stayed 0, progress 1, flags 0x11. Do not walk either of those lines, and do not step on (405,4286) or (410,4285).\n";
prompt += "Cross room 0x80 on a path that stays off those two hit tiles, then continue only while HP stays above 0. Mark AP-082 done only at room 0x12 with $F3C5 == 2, $F3C6 == 0x15, and the follower cleared. If the flail drops HP to 0 again, note the new pose and leave the claim [~]. Do not block AP-082.\n";
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
    summary: "Twenty-second wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
