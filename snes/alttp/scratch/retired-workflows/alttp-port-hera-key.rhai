let meta = #{
    name: "alttp-port-hera-key",
    description: "Take Hera's small key in play and open the big-key chest if a path exists",
    when_to_use: "Continue PORT_PLAN after AP-116. The basement key sprite was reached; the chest was not.",
    phases: [
        #{ title: "Big key", detail: "AP-117 only" },
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
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. A pin is not continuous. Do not overwrite recordings/verified_tip_run.json, Scan_hera_0x77, or Scan_hera_0x87.\n" +
    "You are mgr-hera-key. Your only task is AP-117. AP-116 is done. Do not rescan. Stop before AP-118. Do not start another dungeon. Do not retry AP-101, AP-076, AP-022, AP-064, AP-067, AP-086, AP-093, AP-094, or AP-096. Do not search rooms 0xC9, 0xB9, or 0xA9. Do not load Whatif_EasternPalaceEntry_magic80.\n" +
    "Read docs/routes/hera_ROUTE.md and dungeon/route_data/hera.yaml before you walk. Peg byte $C172: 00 is raised on the HeraEntry west alcove, 01 lowers it. A sword hit on a crystal toggles it in play. Add hit_switch for that measured hit, with one live check. Raised pegs are walls. Do not write $C172.\n" +
    "Small key: sprite 0xE4 at (3712,4512), tile (16,52), room 0x87. At $C172=00 a stand on tile (16,51) already raised keys 0 to 1 in play, hp 24 to 16, pose (3707,4491). Take it again in play and save a new pin. Do not poke the key byte. The saved scan pin still has keys 0.\n" +
    "Big key chest is BigKeyP3 at 0xE9E6, room 0x87 tile (47,52) layer 0. No path from the landing at either peg. yaml requires KeyP3 and canLightTorches. The pin already has lamp $F34A=1 and magic $F36E=0. Lamp use costs 4. If the gap is a torch, a documented whatif_pin.py --set magic=128 on a new Whatif_* copy of the Hera scan pin is allowed. Evidence names field magic, address $F36E, old 0, new 128, and the pin. Do not write that onto an Eastern pin. Do not write $F3CC, rupees, or the big-key bit.\n" +
    "Mark AP-117 done only if the big-key bit rises in play from that chest and the ledger says source play. If two honest tries still have no path, block AP-117 with the peg value, the pose, and what separates the landing from tile (47,52). HP refill at the last heart is allowed and must be disclosed. It is not a Clean claim.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-hera-key", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("hera key manager returned no success");
}

phase("Report");
let report = agent(
    "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and whether AP-117 is x, space, or !. Do not edit files.\n",
    #{ label: "mgr-report", capability_mode: "execute", output_schema: report_schema }
);
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Hera key wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
