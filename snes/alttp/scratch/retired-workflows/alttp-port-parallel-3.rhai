let meta = #{
    name: "alttp-port-parallel-3",
    description: "Boss-room and big-key-room pins for Eastern, Desert, Hera, and Ice",
    when_to_use: "After alttp-port-parallel-2. Do not rewalk the failed corridors.",
    phases: [
        #{ title: "Rooms", detail: "four disclosed room pins" },
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

fn common() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md.\n";
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, HyruleCastleGrounds, Scan_* pins, Play_* pins, or recordings/verified_tip_run.json.\n";
    p += "Read the room id from your dungeon yaml or z3-json-data. Do not invent a room id. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download or commit refs.\n";
    p += "Save the new pin under a Whatif_ name with provenance poke. cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes.\n";
    p += "Read your headings and snes/alttp/AGENTS.md with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes. If you claim a task and the check fails, block it again so the heading does not stay [~]. If you never reach the room, and the task is still [ ], leave it open and add a note. Do not block an unwalked path.\n";
    p += "Do not edit make_entry_pin.py, whatif_pin.py, nav.py, primitives.py, walk_physics.py, hops.py, or another dungeon's files. A scratch probe may live under your recordings directory.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Rooms");
let jobs = [];

let eastern = common();
eastern += "You are the Eastern Armos lane. AP-102 is done. The closed fact is room 0xA9 pose (4856,5592), layer-1 chest (30,26), no layer-0 path. Do not rewalk that chest. eastern.yaml has no other path to Armos.\n";
eastern += "Find the Armos Knights room id. Build one disclosed pin that is indoors in that room, carrying the bow and arrows already proved on Whatif_EasternPalaceEntry_bow. Name it Whatif_Eastern_armos.\n";
eastern += "Then do AP-103 and, if the probe is real, AP-104. Boss step 6 needs at least 10 of 12 RNG offsets. Create snes/alttp/bosses/armos_knights.py only if AP-104's Done line says to. You may edit snes/alttp/docs/routes/eastern_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-103/.\n";
eastern += "If the pin does not load the Armos room, leave AP-103 and AP-104 open and note the pin.\n";
jobs.push(#{ prompt: eastern, label: "mgr-armos", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert Lanmolas lane. Closed facts: room 0x73 shutter and center wall, and Whatif_DesertPalaceEntry_glove has gloves $F354=1 but the rock pile at (704,3168) was not reachable. Do not repeat those walks.\n";
desert += "Find the Lanmolas room id. Build Whatif_Desert_lanmolas indoors in that room. Carry the glove write. Prove the room with pin_probe.\n";
desert += "AP-111 and AP-112 are [!]. Claim them only after the pin is indoors in the Lanmolas room and a worm is on screen. Boss step 6 needs at least 10 of 12 RNG offsets. If the room will not load, note both ids and leave the [!] text in place.\n";
desert += "You may edit snes/alttp/docs/routes/desert_ROUTE.md, snes/alttp/bosses/lanmolas.py if AP-112 requires it, and recordings under snes/alttp/recordings/probe_AP-111/.\n";
jobs.push(#{ prompt: desert, label: "mgr-lanmolas", capability_mode: "all", output_schema: lane_schema });

let hera = common();
hera += "You are the Hera Moldorm lane. Leave AP-118 blocked. Closed fact: room 0x31 slot 0 type 0x1E has no path, and the pit at (47,44) returns to 0x77. Whatif_HeraEntry_pearl has pearl $F357=1 in room 0x77. Do not rewalk those.\n";
hera += "Find the Moldorm room id. Build Whatif_Hera_moldorm indoors in that room. AP-119 and AP-120 are open. Do them only when the pin is in the Moldorm room. Log Link's distance to the boss as the task asks. Boss step 6 needs at least 10 of 12 RNG offsets.\n";
hera += "If the room will not load, leave both tasks open and note the pin. You may edit snes/alttp/docs/routes/hera_ROUTE.md, snes/alttp/bosses/moldorm.py if AP-120 requires it, and recordings under snes/alttp/recordings/probe_AP-119/.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let ice = common();
ice += "You are the Ice big-key lane. AP-165 is [!]. Closed facts: ICE_COAST_PX=18 is already recorded. Rooms 0x0E and 0x1E have no big-key chest. The chest is ROM room 0x1F file 0xE9A4 BigKeyD5. Whatif_Ice_0x1e_bk0 stood on a landing flood that missed the chest, and the switch at (6,56) left the doors shut. Do not repeat that switch or that landing.\n";
ice += "Build Whatif_Ice_0x1f_chest indoors in room 0x1F with big_key 0 and hammer already 1, on the chest tile from the ROM record. Open that chest in play. If the big-key byte rises and the Done line accepts this isolated pin, claim AP-165 as Ice Palace and mark it done. If the byte stays 0, block AP-165 again with the new pose.\n";
ice += "You may edit snes/alttp/docs/routes/ice_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-165/. Do not edit walk_physics.py.\n";
jobs.push(#{ prompt: ice, label: "mgr-ice-chest", capability_mode: "all", output_schema: lane_schema });

let wave = parallel(jobs);
let n = 0;
if wave != () {
    for slot in wave {
        if slot != () && slot.success {
            n += 1;
        }
    }
}
log("room lanes returned " + n.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-103, AP-111, AP-119, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Room-pin wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
