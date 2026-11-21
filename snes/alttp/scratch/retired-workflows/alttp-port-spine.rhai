let meta = #{
    name: "alttp-port-spine",
    description: "Save points, pin audit, and the room 0x50 to 0x72 hop",
    when_to_use: "Continue PORT_PLAN after room_50 is green.",
    phases: [
        #{ title: "Spine tools", detail: "save points and stage replay" },
        #{ title: "Castle hop", detail: "room 0x72, travel, what-if pins" },
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
    p += "Read PORT_PLAN rules once, then only your tasks. Two honest tries, then block. Do not weaken a Done line. Leave the Owner ruling blank.\n";
    p += "ROM commands: cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Do not overwrite recordings/verified_tip_run.json on a red run. No RAM writes on a natural or continuous run. A save-state pin is not continuous.\n";
    p += "Re-read a file immediately before editing it.\n";
    return p;
}

fn lane_prompt(agent, tasks, extra) {
    let p = rules();
    p += "You are manager " + agent + ". Tasks, in order: " + tasks + ".\n";
    p += extra;
    p += "\nSkip a task that is already [x]. If a Needs id is not [x], stop and set waiting_on. Do not block a task whose needs are open.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Spine tools");
let first = [];
first.push(#{
    prompt: lane_prompt(
        "mgr-spine",
        "AP-024, AP-025, AP-026, AP-027",
        "Own run_spine.py, hops.py, tape.py or paths.py as each task lists, and audit_pins.py. room_50 is already green at 13174 frames. Do not edit primitives.py, overworld.py, or rollout.py. Save pins under custom_integrations/Zelda3-Snes/."
    ),
    label: "mgr-spine",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-replay",
        "AP-028, AP-029",
        "Own stage_replay.py and eval_offsets.py. The done edge is room_61 west_to_0x60 from CastleMain. Do not edit hops.py or run_spine.py."
    ),
    label: "mgr-replay",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(first);

phase("Castle hop");
let second = [];
second.push(#{
    prompt: lane_prompt(
        "mgr-castle",
        "AP-073, AP-074",
        "Read snes/alttp/docs/tasks/residual.md before AP-073. Add the natural hops room_50_east_to_0x01 and room_01_down_to_0x72, then 0x72 to 0x82 to 0x81 from the real predecessor. Power-on only. Do not call a pin green continuous. Stop before AP-075. Do not edit primitives.py."
    ),
    label: "mgr-castle",
    capability_mode: "all",
    output_schema: lane_schema,
});
second.push(#{
    prompt: lane_prompt(
        "mgr-travel",
        "AP-030, AP-072",
        "AP-030 owns save_and_quit in primitives.py and a spawn table in ram_map.md. It is a menu action, not an emulator load. AP-072 adds magic costs to the ledger and ram_map.md. What-if pins may own the rods. Do not edit hops.py."
    ),
    label: "mgr-travel",
    capability_mode: "all",
    output_schema: lane_schema,
});
second.push(#{
    prompt: lane_prompt(
        "mgr-whatif",
        "AP-041, AP-042",
        "Own whatif_pin.py and make_entry_pin.py. Never poke $F3CC. If AP-026 is not [x], stop. Delete only the scratch probes the task names."
    ),
    label: "mgr-whatif",
    capability_mode: "all",
    output_schema: lane_schema,
});
second.push(#{
    prompt: lane_prompt(
        "mgr-audit",
        "AP-022",
        "AP-022 is blocked on an older red tip that reported set_state=8. The tip is now green: run_to_verified_tip.py --ledger, room_50, 13174 frames, writes 0. Re-measure set_state and writes on that green run. If both are 0, mark AP-022 done. If set_state is still the startup inject or the bush restores, leave it blocked and replace the reason with the new numbers. Do not delete those restores to force a zero."
    ),
    label: "mgr-audit",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(second);

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python.\n";
report_prompt += "A heading matches a line starting with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !.\n";
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
    summary: "Spine wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
