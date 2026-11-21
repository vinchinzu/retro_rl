let meta = #{
    name: "alttp-port-magic",
    description: "Refill Eastern magic in play and open the big key chest",
    when_to_use: "Continue PORT_PLAN after AP-101 blocked on magic 0 with the lamp equipped.",
    phases: [
        #{ title: "Magic", detail: "one Eastern big-key attempt" },
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

phase("Magic");
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. No RAM writes. A pin is not continuous.\n" +
    "You are mgr-magic. The only task is AP-101.\n" +
    "EasternPalaceEntry has lamp $F34A=1 and magic $F36E=0x00. Lamp use on that pin costs 4. The big key chest in eastern.yaml requires hasLightCone. There is no small key in that file. The Cannonball Chest already paid 100 rupees; do not open it again.\n" +
    "In rooms 0xC9, 0xB9, and 0xA9 only, find a magic drop and pick it up in play. Then light the cone and open the big key chest. The big key bit must rise in play. If those three rooms have no magic drop, block AP-101 with the rooms you searched. Do not poke magic. Do not start Desert. Do not walk the 0xB9 east door.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-magic", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("magic manager returned no success");
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
    summary: "Magic wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
