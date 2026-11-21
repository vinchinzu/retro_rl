let meta = #{
    name: "alttp-port-castle",
    description: "Correct the 0x81 door check, then the big key and the ready side tasks",
    when_to_use: "Continue PORT_PLAN once power-on reaches room 0x81.",
    phases: [
        #{ title: "Door and side tasks", detail: "AP-035, bottle, push, overworld, realtime" },
        #{ title: "Big key", detail: "trooper and the gold lock" },
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
    p += "Read the PORT_PLAN rules once, then only your tasks. Two honest tries, then block. Leave the Owner ruling blank.\n";
    p += "ROM commands: cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Do not overwrite recordings/verified_tip_run.json on a red run. No RAM writes on a natural or continuous run. A pin is not continuous.\n";
    p += "Re-read a file immediately before editing it. z3-json-data and the live scan beat a hint.\n";
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

phase("Door and side tasks");
let first = [];
first.push(#{
    prompt: lane_prompt(
        "mgr-doors",
        "AP-035, AP-063",
        "AP-035 is blocked because its Done line asks room 0x81 for a west door type. The live slots and the ROM door list have only a north slot, type 0x40. The gold lock is object 0x18 on room 0x80, not a door slot. Replace that Done line with: pin_probe --doors prints each real slot as direction, type, and destination, and the 0x81 report names the 0x80 object lock instead of a west door. Implement that. Do not invent a west type. Then do AP-063. clear_room runs only where the door table says a clear is required."
    ),
    label: "mgr-doors",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-prim",
        "AP-066, AP-067",
        "AP-066: use whatif_pin.py to make a red-potion pin. Drink in play. The ledger must show the bottle spend. AP-067 is blocked only on push_block. Bomb, lift, and throw already have live checks; do not redo them. Find a room whose object list has a state-0 block on attribute 0x70 from z3-json-data, and push that block. EasternPalaceEntry's 0x7N tiles are pots. If no reachable room has a resting block, block AP-067 and name the room id."
    ),
    label: "mgr-prim",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-ow",
        "AP-084",
        "AP-084 is blocked because no pin stands on Sanctuary screen 0x13. AP-030 measured Save and Quit: progress 03, menu row 1, sanctuary in 669 frames. Use that menu action to land outdoors, save the resulting state, and walk Sanctuary exit to Link's house screen. Record the provenance. Do not call the walk continuous. If the landing screen is not 0x13, block and write the screen id you got."
    ),
    label: "mgr-ow",
    capability_mode: "all",
    output_schema: lane_schema,
});
first.push(#{
    prompt: lane_prompt(
        "mgr-rt",
        "AP-198",
        "Read the stage reports already on disk. List every stage under 1x real time. Apply the hazard gate once before shortening the horizon. The Done line allows a table of the stages that are still under 1x. Write that table and stop. Do not retune every room."
    ),
    label: "mgr-rt",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(first);

phase("Big key");
let castle = agent(lane_prompt(
    "mgr-castle",
    "AP-075, AP-076",
    "Power-on already reaches room 0x81 in 17327 frames (opening_spine_room_81.json). AP-075 kills the ball-and-chain trooper, sprite 0x6A, with strike_if_safe, and the big-key bit must rise in play. AP-076 opens the gold lock. It is object 0x18 on room 0x80, not a 0x81 west door type. Isolate west_to_0x80. Stop before AP-077. If AP-035 or AP-063 is not [x], stop and set waiting_on."
), #{ label: "mgr-castle", capability_mode: "all", output_schema: lane_schema });
if castle == () || castle.success != true {
    log("castle manager returned no success");
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
    summary: "Castle wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
