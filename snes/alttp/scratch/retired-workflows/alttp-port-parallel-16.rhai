let meta = #{
    name: "alttp-port-parallel-16",
    description: "Step from the Desert plaque's side cells onto the two unread 0x46 tiles",
    when_to_use: "After alttp-port-parallel-15. The y=3464 cells were beside the stand, not under it. Kholdstare stays claimed.",
    phases: [
        #{ title: "Plaque", detail: "two side steps" },
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

phase("Plaque");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet. Do not claim AP-112, AP-114, or AP-168.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md, moldorm.py, or kholdstare.py.\n";
prompt += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory.\n";
prompt += "Closed: from (304,3472) with book 1, UP stays there and $02ED stays 0. LEFT then UP stops at (296,3472) and does not enter (296,3464). An east step stops at (312,3464). A west step stays at (280,3464). North from the pad (296,3259) stops in sand at (296,3432). Do not repeat those approaches.\n";
prompt += "From (312,3464), step west onto (304,3464). From (280,3464), step east onto (296,3464). Read $02ED on each cell. Mark AP-113 done only if that flag rises in play and the Done line passes. If both cells refuse the step or leave $02ED at 0, note the poses and leave the claim [~]. Do not block AP-113.\n";
prompt += "You may edit snes/alttp/docs/routes/desert_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-113/.\n";
prompt += "Return completed, waiting_on, and blocked.\n";
let lane = agent(prompt, #{
    label: "mgr-tablet",
    capability_mode: "all",
    output_schema: lane_schema,
});
if lane == () || lane.success != true {
    log("tablet lane returned no success");
} else {
    log("tablet lane returned");
}

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-113 and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Sixteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
