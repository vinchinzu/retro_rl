let meta = #{
    name: "alttp-port-parallel-17",
    description: "Dash once from each side of the Desert plaque onto the unread 0x46 cells",
    when_to_use: "After alttp-port-parallel-16. Walks from (312,3464) and (280,3464) stay put. Kholdstare stays claimed.",
    phases: [
        #{ title: "Dash", detail: "one dash from each side cell" },
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

phase("Dash");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet. Do not claim AP-168 or edit kholdstare.py or moldorm.py.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md.\n";
prompt += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell.\n";
prompt += "Walks are closed. From (312,3464) book 1, LEFT stays there, feet attribute 0x00, and does not enter (304,3464). From (280,3464), RIGHT stays there and does not enter (296,3464). $02ED stayed 0. Do not hold LEFT, RIGHT, or UP again.\n";
prompt += "Dash is already measured: charge 30 frames, then state 17 at 4 px/frame. One dash west from (312,3464). One dash east from (280,3464). Read the pose and $02ED after each. Mark AP-113 done only if the flag rises in play and the Done line passes. If both dashes stay on the side cells, note the poses and leave the claim [~]. Do not block AP-113.\n";
prompt += "You may edit recordings under snes/alttp/recordings/probe_AP-113/. Return completed, waiting_on, and blocked.\n";
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the lowest open id, and the box mark for AP-113 and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Seventeenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
