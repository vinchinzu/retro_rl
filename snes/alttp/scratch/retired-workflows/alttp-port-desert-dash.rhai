let meta = #{
    name: "alttp-port-desert-dash",
    description: "Test one documented dash-ability write on the Desert torch, then stop",
    when_to_use: "Continue PORT_PLAN after AP-109 blocked because boots are owned and the dash bit is clear.",
    phases: [
        #{ title: "Dash bit", detail: "one what-if, one torch try" },
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
let work = agent(
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py note. Do not run bd. Do not git add or commit. cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. A pin is not continuous. Do not overwrite recordings/verified_tip_run.json, Scan_desert pins, or DesertPalaceEntry.\n" +
    "You are mgr-desert-dash. AP-109 stays blocked. Do not mark AP-109, AP-089, AP-101, or AP-117 done. Do not clear a [!] line. Do not start another dungeon. Do not retry the Eastern magic what-if or the Hera wall. Do not search rooms 0xC9, 0xB9, or 0xA9.\n" +
    "Measured fact: DesertPalaceEntry boots $F355=1, ability $F379=0xF8, and the probe treated bit 0x04 as the dash bit. A+UP and A+RIGHT at sprite 0x3B (1856,3680) in room 0x73 left link_state 0 and keys 0. Pose after A+UP was (1872,3638). Pose after A+RIGHT was (1935,3664). Room 0x75 chest BigKeyP2 tile (47,10) stayed closed. The center wall is cols 27-36. The 0x74 north-lip pot already raised magic 0 to 16. Do not repeat it.\n" +
    "Read the dash check before you write a byte. Confirm which address and which bit the boots dash reads. The probe's bit 0x04 is a candidate until that read agrees. whatif_pin.py --set only accepts ram.INVENTORY_FIELDS names. Boots is already 1, so --set boots=1 is not a new result. If the dash bit is not a named field, add one InventoryField for the verified address in ram.py and cover it in the existing inventory test. Do not add a raw-address flag to whatif_pin.py. Do not write $F3CC, keys, rupees, HP, or the big-key bit.\n" +
    "Make one Whatif_* copy of DesertPalaceEntry with that dash bit set. Provenance kind whatif. Evidence names the field, the address, the old value, and the new value. Walk to sprite 0x3B and press the dash. Two honest tries. If keys rise, save a new pin and try the big-key chest in 0x75 once. The ledger must show the ability change as a poke. A poke cannot satisfy the AP-109 done line.\n" +
    "Record the result with: python3 /home/v/.grok/alttp_port_claim.py note AP-109 \"...\" Include whether keys rose and whether the big-key word rose. HP refill at the last heart is allowed and must be disclosed. It is not a Clean claim. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n" +
    "Return completed, waiting_on, and blocked. completed stays empty unless you marked a different task done, which you should not.\n",
    #{ label: "mgr-desert-dash", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("desert dash manager returned no success");
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
    summary: "Desert dash wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
