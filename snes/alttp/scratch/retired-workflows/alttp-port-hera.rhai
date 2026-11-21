let meta = #{
    name: "alttp-port-hera",
    description: "Scan Hera, and acquire the lamp and magic on the earlier tickets",
    when_to_use: "Continue PORT_PLAN after AP-115 is green and the Eastern cone what-if is blocked.",
    phases: [
        #{ title: "Hera and lamp", detail: "AP-116 scan, plus lamp and magic from the earlier rooms" },
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
    p += "Ticks only through python3 /home/v/.grok/alttp_port_claim.py (claim, done, block, note). Leave the Owner ruling blank.\n";
    p += "ROM commands: cwd snes/, SDL_VIDEODRIVER=dummy. Pytest: QT_QPA_PLATFORM=offscreen. One emulator in your process. Do not kill other Python processes.\n";
    p += "A pin is not continuous. Do not overwrite recordings/verified_tip_run.json on a red run. Do not overwrite an existing Scan_ pin.\n";
    p += "Do not write $F3CC, keys, the big-key bit, or rupees. Do not retry AP-076, AP-022, AP-064, AP-067, AP-086, AP-093, AP-094, AP-096, or AP-101.\n";
    p += "Do not search rooms 0xC9, 0xB9, or 0xA9 for magic. Do not load Whatif_EasternPalaceEntry_magic80 and do not make another Eastern magic what-if. The layer-1 wall at x=4672 already blocked that chest.\n";
    p += "z3 route yaml and a live scan beat a hint. Re-read a file immediately before editing it. Soft max about 1000 lines; merge into the owning module.\n";
    return p;
}

fn hera_prompt() {
    let p = rules();
    p += "You are mgr-hera. Your only task is AP-116, the Hera scan. AP-115 is done. Do not repeat the peg-byte hop as a new task. Stop before AP-117. Do not start another dungeon.\n";
    p += "Pin HeraEntry is room 0x77. Crystal toggles peg byte $C172. west_down_to_0x87 already landed room 0x87 at (3721,4424), 2/2.\n";
    p += "Read dungeon/route_data/hera.yaml and docs/routes/DUNGEON_TEMPLATE.md. Copy the Eastern scan shape into docs/routes/hera_ROUTE.md. Walk only rooms reachable with the keys the pin holds. Save Scan_hera_<room> with provenance kind scan. pin_probe doors, chests, and tiles. Door-only maps/room_XX.json skeletons. Record $C172 in each room row. Name every stopping lock. Hitting a crystal in play to pass a peg is allowed. A RAM write of the peg byte is not.\n";
    p += "Mark AP-116 done only when every reachable room is in the table, each stopping lock is named, and room_engine lint is green. If a lock stops you, that lock is the scan result, not a reason to block the scan.\n";
    p += "HP refill at the last heart is allowed on this dev scan. Disclose assist on, write count, and hearts in the evidence line. That refill is not a Clean claim. Do not write the lamp or magic on this lane; the other lane owns those bytes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

fn lamp_prompt() {
    let p = rules();
    p += "You are mgr-lamp. Go back to the earlier rooms and get the lamp and magic. Do not open the Eastern big-key chest. Do not edit hera_ROUTE.md, Hera maps, or AP-116.\n";
    p += "Play first. The unopened lamp chests are room 0x104 Link's house (Scan_0x0104, item Lamp, z3 0xE9BC) and room 0x55 secret passage (Scan_0x55, item Lamp, z3 0xE971). Open one in play from its Scan pin. Save a new pin. Provenance kind resumed. Ledger must show the lamp rising with source play. Two honest tries, then stop walking that chest.\n";
    p += "Magic is $F36E, inventory field magic. A Desert room 0x74 pot already raised magic 0 to 16; do not repeat that pot as a new result. Find a different drop outside 0xC9, 0xB9, and 0xA9, and take it in play. Save a new pin, kind resumed. Two honest tries.\n";
    p += "If play does not raise the byte, a documented write is allowed. Use whatif_pin.py --set lamp=1 or --set magic=128 on a new Whatif_* copy of a pin that is not EasternPalaceEntry. The sidecar kind is whatif. Evidence names the field, the address, the old value, and the new value. Do not walk that pin into Eastern. Delete nothing that another lane saved.\n";
    p += "HP refill at the last heart is allowed. Disclose it. It does not write inventory. Read AP-068. Mark AP-068 done only when a Survival run of the existing tip reports the refill count and the audit counts those writes. AP-022 stays blocked; do not retry it. If that done line is not met, leave AP-068 open and set waiting_on to the audit fact.\n";
    p += "Record the lamp result with: python3 /home/v/.grok/alttp_port_claim.py note AP-065 \"...\" and the magic result with note AP-072. Do not mark AP-065 or AP-072 done again. Do not note AP-101.\n";
    p += "Return completed, waiting_on, and blocked. completed lists only tasks you marked done.\n";
    return p;
}

phase("Hera and lamp");
let jobs = [];
jobs.push(#{
    prompt: hera_prompt(),
    label: "mgr-hera",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: lamp_prompt(),
    label: "mgr-lamp",
    capability_mode: "all",
    output_schema: lane_schema,
});
let results = parallel(jobs);
let n = 0;
for r in results {
    if r == () || r.success != true {
        log("lane " + n.to_string() + " returned no success");
    }
    n += 1;
}

phase("Report");
let report = agent(
    "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, and the lowest open id. Also say whether AP-116 is x, space, or !. Do not edit files.\n",
    #{ label: "mgr-report", capability_mode: "execute", output_schema: report_schema }
);
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Hera wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
