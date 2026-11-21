let meta = #{
    name: "alttp-port-scan",
    description: "Close the Eastern scan on the rupee chest, then scan Desert",
    when_to_use: "Continue PORT_PLAN after the 0xB9 chest paid rupees and Desert step 1 is green.",
    phases: [
        #{ title: "Scans", detail: "Eastern table, then Desert" },
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

phase("Scans");
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. No RAM writes. A pin is not continuous.\n" +
    "You are mgr-scan. Tasks, in order: AP-100, AP-101, AP-108.\n" +
    "The 0xB9 chest at tile (50,37) xy=(5009,5928) opened in play and paid rupees 0 to 100. Keys stayed 0. That chest is eastern.yaml Cannonball Chest, item OneHundredRupees, address 0xE9B3. Do not open it again.\n" +
    "eastern.yaml has no small key. The big key chest requires hasLightCone. Door type 0x1C on 0xB9 east was treated as a small-key door; check the live door and the yaml before assuming a key.\n" +
    "AP-100's Done line is already: every room reachable with the keys you hold is in the table, each stopping lock is named, lint green. Fill that table from the existing Scan dumps for the rooms already reached. Name the locks. Mark AP-100 done when that is true. Do not walk a door you cannot open.\n" +
    "AP-101: read the Eastern entry pin's lamp byte. The big key rises in play only if the lamp can light the cone. If the pin has no lamp, block AP-101 with the lamp value and do not hunt a small key. If the lamp is present, get the big key from the Big Key Chest in play.\n" +
    "AP-108 is the Desert scan. Step 1 is green: room 0x84 north_to_0x74, landing (2272,4052). Scan rooms reachable with the keys you hold, record the outdoor link if you can walk it, name any stopping lock. Stop before AP-109.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-scan", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("scan manager returned no success");
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
    summary: "Scan wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
