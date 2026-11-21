let meta = #{
    name: "alttp-port-desert",
    description: "Take Desert's torch key in play and open the big-key chest if a path exists",
    when_to_use: "Continue PORT_PLAN after AP-117 blocked. Desert steps 1 and 2 are green. Eastern and Hera big keys stay blocked.",
    phases: [
        #{ title: "Big key", detail: "AP-109 only" },
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

phase("Big key");
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. A pin is not continuous. Do not overwrite recordings/verified_tip_run.json or the Scan_desert pins.\n" +
    "You are mgr-desert. Your only task is AP-109. AP-107 and AP-108 are done. Do not rescan those two rooms as a new task. Stop before AP-110. Do not start another dungeon. Do not retry AP-101, AP-117, AP-076, AP-022, AP-064, AP-067, AP-086, AP-093, AP-094, or AP-096. Do not search rooms 0xC9, 0xB9, or 0xA9. Do not load Whatif_EasternPalaceEntry_magic80. Do not mark AP-089 done. Do not edit hera_ROUTE.md or eastern_ROUTE.md.\n" +
    "Read docs/routes/desert_ROUTE.md and dungeon/route_data/desert.yaml before you walk. Re-read a file immediately before you edit it. Soft max about 1000 lines; merge into the owning module.\n" +
    "Scan facts, keys held 0: rooms 0x84 and 0x74 only. DesertPalaceEntry room 0x84, north_to_0x74 landed (2272,4052) 2/2. Boots $F355=1, lamp=1, gloves=0, magic=0 on the scan pins. The 0x74 north-lip pot already raised magic 0 to 16. Do not repeat that pot. 0x27 statues seal the center pocket. Both 0x74 mouths are internal type 0x00. Outdoor back is screen 0x30 and was not walked. Gloves are 0, so do not lift rocks as the key path.\n" +
    "Torch key is KeyP2, yaml address 0x180160, region Desert Palace Main (Outer), requires canDash. Big key chest is BigKeyP2, address 0xE9C2, region Desert Palace East, requires KeyP2. Neither room was entered on the scan. Take the torch key in play. Boots are already 1, so a dash is a button press on that pin, not the AP-089 task. Do not poke the key byte, the big-key bit, $F3CC, or rupees. Save a new pin when the key rises. Provenance kind resumed. Ledger must say source play.\n" +
    "Then walk to the big-key chest with that key and open it in play. Mark AP-109 done only if the big-key bit rises in play and the ledger says source play. If two honest tries still have no path, block AP-109 with the room, the pose, and what separates the walker from the torch and from address 0xE9C2. HP refill at the last heart is allowed and must be disclosed. It is not a Clean claim. z3 route yaml and a live scan beat a hint.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-desert", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("desert manager returned no success");
}

phase("Report");
let report = agent(
    "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and whether AP-109 is x, space, or !. Do not edit files.\n",
    #{ label: "mgr-report", capability_mode: "execute", output_schema: report_schema }
);
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Desert wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
