let meta = #{
    name: "alttp-port-ability-play",
    description: "See if DesertPalaceEntry sets the dash bit in play, then stop",
    when_to_use: "Continue PORT_PLAN after the Desert ability poke dashed and the key stayed 0. No later dungeon is ready.",
    phases: [
        #{ title: "Dash bit", detail: "writers only, then one play trigger" },
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
    "Repo: /home/v/01_projects/11_games/retro_rl. Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py note. Do not run bd. Do not git add or commit. cwd snes/ and SDL_VIDEODRIVER=dummy if you boot a ROM. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. A pin is not continuous. Do not overwrite recordings/verified_tip_run.json, Scan_desert pins, DesertPalaceEntry, or Whatif_DesertPalaceEntry_abilityFC.\n" +
    "You are mgr-ability-play. AP-109 stays blocked. Do not mark AP-109, AP-089, AP-101, or AP-117 done. Do not clear a [!] line. Do not start another dungeon. Do not claim AP-068 or AP-022. Do not retry the Eastern magic what-if or the Hera wall. Do not search rooms 0xC9, 0xB9, or 0xA9. Do not load Whatif_DesertPalaceEntry_abilityFC. Do not call whatif_pin.py. Do not write $F379, $F3CC, keys, rupees, HP, or the big-key bit.\n" +
    "Measured fact, already on the AP-109 note: Whatif_DesertPalaceEntry_abilityFC set ability $F379 from 0xF8 to 0xFC. The dash routine at $07:9C2F ANDs that byte with mask 0x04. The ledger booked the change as write frame 1 source=write, a poke. At sprite 0x3B in room 0x73, A+UP went (1866,3715) to (1872,3640) with link_state {0,17}, and A+RIGHT went (1850,3664) to (1934,3664) with link_state {0,17}. Keys stayed 0. The big-key word stayed 0x0000 and the room 0x75 chest was not tried. One last-heart refill, hp 4 to 24, writes=1. A poke cannot close AP-109. Do not press A at sprite 0x3B again.\n" +
    "Boots receipt Y=$4B ORs bit 0x04 at $09:8654 and writes $F355=1. DesertPalaceEntry already has boots $F355=1 and ability $F379=0xF8. Read the ROM, or a disassembly already on this machine, and list every store that can set bit 0x04 of $7EF379. Name the address and the caller. Use read_file, grep, and the shell. Do not answer from memory.\n" +
    "If the only such store is the boots receipt, and that receipt cannot run on DesertPalaceEntry because boots are already 1, do not boot the ROM. Record the writer list with: python3 /home/v/.grok/alttp_port_claim.py note AP-109 \"...\". Say the bit cannot rise in play on this pin.\n" +
    "Boot the ROM only if you find a writer the game itself runs while boots are already owned, such as a room transition or an item menu, and that writer is not a RAM poke. Try that trigger once on DesertPalaceEntry. The ledger must show source=play for any change to $F379. If the bit stays clear, stop and note the trigger and the byte. If the bit rises in play, walk the torch interaction that writer implies, two honest tries. Do not repeat the sprite 0x3B A-press. If keys stay 0, do not open the 0x75 chest. Mark nothing done. HP refill at the last heart is allowed and must be disclosed. It is not a Clean claim. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n" +
    "Return completed, waiting_on, and blocked. completed stays empty.\n",
    #{ label: "mgr-ability-play", capability_mode: "all", output_schema: lane_schema }
);
if work == () || work.success != true {
    log("ability-play manager returned no success");
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
    summary: "Ability-play wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
