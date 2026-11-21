let meta = #{
    name: "alttp-port-chest",
    description: "Open the Eastern 0xB9 chest, then the Desert entry if it stays shut",
    when_to_use: "Continue PORT_PLAN after the Eastern scan stopped on the layer-1 chest.",
    phases: [
        #{ title: "Chest", detail: "0xB9 chest, then Desert entry" },
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

phase("Chest");
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. No RAM writes. A pin is not continuous.\n" +
    "You are mgr-chest. Tasks: AP-100, then AP-107 only if AP-100 is still blocked.\n" +
    "AP-100 is blocked. Pots in 0xC9 and 0xA9 opened in play and keys stayed 0. The 0xB9 chest at tile (50,36) and its jars stayed closed. Layer-1 landings at (4952,5754) and (4808,5754) are not the chest. South ledges stopped on layer 0 at (4847,6056) and (4916,6057). Read recordings/scan/Scan_eastern_0xb9.txt and stand on the chest tile on its own layer, then press the open button. Two tries. If a key rises in play, walk the 0xB9 east door (type 0x1C) and finish the scan under the current Done line: reachable rooms tabulated, stopping locks named, lint green. If the chest stays closed, leave AP-100 blocked with the pose you stood on, and do AP-107.\n" +
    "AP-107 is Desert only: DesertPalaceEntry room 0x84, forward edge north_to_0x74, isolated, not natural. Stop before AP-108.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-chest", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("chest manager returned no success");
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
    summary: "Chest wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
