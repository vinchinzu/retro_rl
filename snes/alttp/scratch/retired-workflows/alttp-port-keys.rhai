let meta = #{
    name: "alttp-port-keys",
    description: "Finish the Eastern scan at the small-key door, then the Desert entry hop",
    when_to_use: "Continue PORT_PLAN after Eastern step 1 is green and the scan stopped on a locked door.",
    phases: [
        #{ title: "Keys", detail: "Eastern scan and big key, then Desert entry" },
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
    p += "Two honest tries, then block. Leave the Owner ruling blank. A pin is not continuous. No RAM writes on a natural run. Never poke $F3CC.\n";
    p += "ROM commands: cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Re-read a file immediately before editing it. Add hop rows; do not rewrite the hop table. z3 route_data beats a hint.\n";
    return p;
}

phase("Keys");
let work = agent(rules() + "You are manager mgr-keys. Tasks, in order: AP-100, AP-101, AP-107, AP-108.\n" +
    "Eastern step 1 is green: room_c9 north_to_0xb9, 2/2, landing (4856,6104). AP-100 saved Scan_eastern_0xc9 and Scan_eastern_0xb9, then stopped because 0xB9 east is small-key type 0x1C and keys were 0. North had no path. West LEFT from (4672,5764) stayed in 0xB9.\n" +
    "The scan Done line asks for every route room, and the big-key route that finds the key is the next task. Change the template step-2 Done sentence so a scan is done when every room reachable with the keys you hold is in the table, each stopping lock is named, and lint is green. Do not pretend a locked room was scanned.\n" +
    "Before you accept keys=0, open the chests and pots in 0xC9 and 0xB9 in play. If a key rises, walk through that door and keep scanning. Then mark AP-100 done only if the reachable rooms and the named lock satisfy the corrected line.\n" +
    "AP-101 then isolates the edges you can reach and picks up the big key in play. Keys come from play only. Stop before AP-102 if the big key does not rise.\n" +
    "Then Desert only. Pin DesertPalaceEntry room 0x84, forward edge north_to_0x74. AP-107 is the entry hop, isolated, not natural. AP-108 scans rooms reachable with the keys you hold, including the outdoor link between the front and back if you can walk it. Stop before AP-109.\n" +
    "Skip a task that is already [x]. If a Needs id is not [x], stop and set waiting_on.\n" +
    "Return completed, waiting_on, and blocked.\n",
    #{ label: "mgr-keys", capability_mode: "all", output_schema: lane_schema });
if work == () || work.success != true {
    log("keys manager returned no success");
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
    summary: "Keys wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
