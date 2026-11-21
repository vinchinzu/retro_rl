let meta = #{
    name: "alttp-port-cone",
    description: "Light the Eastern cone from a disclosed magic pin, or start Hera",
    when_to_use: "Continue PORT_PLAN after the three Eastern rooms had no magic drop.",
    phases: [
        #{ title: "Cone", detail: "what-if magic, then Hera only if the chest stays shut" },
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

phase("Cone");
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. A pin is not continuous. Never poke $F3CC or the big-key bit.\n" +
    "You are mgr-cone. Tasks: AP-101, then AP-102 only if the big key rose. If AP-101 stays blocked, do AP-115 and stop.\n" +
    "Rooms 0xC9, 0xB9, and 0xA9 have no magic drop (no sprite 0xDF or 0xE0). Pots and popos dropped 0xD9 and 0xDA. 0xA9 pots dropped 0xD8. Do not search those rooms for magic again.\n" +
    "Dungeon steps 1-6 are dev. Make a disclosed what-if from EasternPalaceEntry with lamp $F34A left at 1 and magic $F36E at least 4. Record the poke in the pin provenance. Use whatif_pin.py if it can set that byte; otherwise write the provenance the same way AP-041 did. Then light the cone and open the Big Key Chest in play. Mark AP-101 done only if the big key bit rises in the ledger from that chest.\n" +
    "If it rises, do AP-102: the bow from the big chest, and a shoot_arrow primitive with a live check. Stop before AP-103.\n" +
    "If the chest does not open, leave AP-101 blocked with the magic value you set and the pose at the chest. Then do only AP-115. Pin HeraEntry is room 0x77. Hit a switch, name the peg-state byte, add it to ram.py, and isolate the first hop. Stop before AP-116.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-cone", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("cone manager returned no success");
}

phase("Report");
let report = agent(
    "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, and the lowest open id. Do not edit files.\n",
    #{ label: "mgr-report", capability_mode: "execute", output_schema: report_schema }
);
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Cone wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
