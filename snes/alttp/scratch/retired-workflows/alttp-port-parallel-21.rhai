let meta = #{
    name: "alttp-port-parallel-21",
    description: "Replay the Clean secret hole, then cross room 0x80 without an HP refill",
    when_to_use: "After alttp-port-parallel-20. Room 0x55 loads from (2431,1704). The escort dies in room 0x80.",
    phases: [
        #{ title: "Escort", detail: "room 0x80 without a refill" },
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

phase("Escort");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are the Sanctuary Clean lane. Claim name is exactly sanctuary-clean. AP-082 is [~] sanctuary-clean. Do not claim AP-113 or AP-168.\n";
prompt += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add or commit. Do not edit docs/STATUS.md.\n";
prompt += "The hole is solved. Facing-up A samples (link_x+7, link_y+6) via ROM $07:D365 and $07:D36D when $2F=0. From screen 0x1B pose (2431,1704), the probe (2438,1710) is attribute 0x50 map16 0x36, and SECRET_HOLE_ENTRY_SCRIPT enters room 0x55 at (2975,2710) with HP 24. No search set_state and no refill. Do not miss that bush again.\n";
prompt += "The escort then died in room 0x80 at (303,4232), HP 0, mode 0x12, during room_81_to_zelda. Progress was 1, flags 0x11, follower 0. Do not repeat that death line.\n";
prompt += "Power on again. No load, no RAM write, no HP refill, no search restore. Enter 0x55 by the aligned bush, then cross room 0x80 by a different path. Mark AP-082 done only at room 0x12 with $F3C5 == 2, $F3C6 == 0x15, and the follower cleared. If you die in 0x80 again, note the new pose and HP and leave the claim [~]. Do not block AP-082.\n";
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
    summary: "Twenty-first wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
