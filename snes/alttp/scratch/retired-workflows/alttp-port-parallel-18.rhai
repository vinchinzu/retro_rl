let meta = #{
    name: "alttp-port-parallel-18",
    description: "Put the receipt dash bit on the Desert plaque pin and dash once from each side cell",
    when_to_use: "After alttp-port-parallel-17. A+direction on ability 0xF8 never entered state 17.",
    phases: [
        #{ title: "Dash bit", detail: "one dash from each side cell" },
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

phase("Dash bit");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet. Do not claim AP-168 or edit kholdstare.py or moldorm.py.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md.\n";
prompt += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell.\n";
prompt += "Closed: A+LEFT from (312,3464) and A+RIGHT from (280,3464) stayed put. Book 1, boots 1, ability $F379=0xF8, dash bit 0, state stayed 0, $5E stayed 0, $02ED stayed 0, feet attribute 0x00. State 17 never started. Do not press A on ability 0xF8 again. Do not search for another writer of bit 0x04.\n";
prompt += "The Sahasrahla receipt already set ability $F379 from 0xF8 to 0xFC in play. Copy the screen 0x30 pin to Whatif_Desert_plaque and set ability to 0xFC. Record that as a poke. Keep book 1. Prove the byte with pin_probe.\n";
prompt += "One dash west from (312,3464). One dash east from (280,3464). Read state, pose, and $02ED after each. Mark AP-113 done only if $02ED rises in play and the Done line passes. If state 17 never starts, or it starts and both poses stay on the side cells, note that and leave the claim [~]. Do not block AP-113.\n";
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
    summary: "Eighteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
