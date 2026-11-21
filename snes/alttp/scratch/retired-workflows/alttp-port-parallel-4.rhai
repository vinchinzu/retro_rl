let meta = #{
    name: "alttp-port-parallel-4",
    description: "Ice chest ability bit, Desert room 0x33 load, Moldorm placed hit, Armos arrow after spawn",
    when_to_use: "After alttp-port-parallel-3. Do not rewalk the failed corridors or the failed boss policies.",
    phases: [
        #{ title: "Lanes", detail: "four new facts" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, HyruleCastleGrounds, Scan_* pins, Play_* pins, Whatif_Eastern_armos, Whatif_Hera_moldorm, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read your files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes. If you claim a blocked task and the check fails, block it again so the heading does not stay [~].\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Ice chest lane. AP-165 is [!]. Closed fact: Whatif_Ice_0x1f_chest is room 0x1F at (7528,944) tile (45,54). Facing-up A on the BigKeyD5 cells left opened=0 and big_key 0. Ability $F379=0x1E misses the chest A-press mask 0xE0. Do not repeat that A press on ability 0x1E.\n";
ice += "Read the chest routine that tests mask 0xE0. On a copy named Whatif_Ice_0x1f_open, set only the ability bit that routine requires. Keep every other ability bit. Prove the byte with pin_probe. Press A once on tile (45,54). If the big-key byte rises and the Done line accepts this isolated pin, claim AP-165 as Ice Palace and mark it done. If it stays 0, block AP-165 again with the new ability byte and the chest opened flag.\n";
ice += "You may edit snes/alttp/docs/routes/ice_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-165/. Do not edit walk_physics.py or another dungeon's files.\n";
jobs.push(#{ prompt: ice, label: "mgr-ice-open", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert Lanmolas lane. Closed facts: entrance 0x0C loads room 0x63, not 0x33. The north stair type 0x20 stays behind attribute 0xF0. Screen 0x30 never reached the rock at (704,3168). Whatif_Desert_lanmolas was not saved. Do not walk entrance 0x0C, the push-block, or screen 0x30.\n";
desert += "Read snes/alttp/recordings/probe_AP-103/load_armos.py. Copy that warp into your own script under snes/alttp/recordings/probe_AP-111/. Do not edit load_armos.py. Source pin is an indoor Desert pin (room 0x84). Target room is 0x33. Warp submodule 0x15, previous room kept, $A0 set to 0x33, never write $F3CC. Save Whatif_Desert_lanmolas only when pin_probe shows indoors room 0x33 and sprite 0x54 is on screen.\n";
desert += "Then claim AP-111 and run the probe. Boss step 6 for AP-112 needs at least 10 of 12 RNG offsets, and only after a worm is on screen. You may edit snes/alttp/docs/routes/desert_ROUTE.md and snes/alttp/bosses/lanmolas.py. If room 0x33 does not load, note AP-111 and leave the existing [!] text in place.\n";
jobs.push(#{ prompt: desert, label: "mgr-lanmolas", capability_mode: "all", output_schema: lane_schema });

let hera = common();
hera += "You are the Moldorm fight lane. AP-119 is done. AP-120 is open. Do not touch AP-118.\n";
hera += "Closed fact on Whatif_Hera_moldorm, room 0x07: a placed strike drops HP $0E50 from 12 to 10 at 8-16 px east facing LEFT or south facing UP. Walked rel=(-4,0) facing UP previews a whiff and must not be repeated. A north-rim probe loaded room 0x17; that is a floor hit.\n";
hera += "Read snes/alttp/bosses/moldorm.py. Commit swings only from the placed-hit window. Score the offsets the Done line requires. Mark AP-120 done only when the heart container is taken on at least 10 of 12 offsets. If the heart stays untaken, leave AP-120 open and note the offset count. Do not block it for the already recorded whiff.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py, snes/alttp/docs/routes/hera_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-119/. Do not edit armos_knights.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let eastern = common();
eastern += "You are the Armos fight lane. AP-103 is done. AP-104 is [!]. Closed facts on Whatif_Eastern_armos, room 0xC8: six knights, HP $0E50=48. An arrow drops HP by 16 only after the knight leaves its spawn xy. Spawn-pose arrows halt as ancilla 0x0A and leave HP at 48. The south-line fight through frame 7000 left the knights alive and the heart untaken. Do not repeat that south-line fight or those spawn-pose arrows.\n";
eastern += "Read snes/alttp/dungeon/bosses/armos_knights.py. New policy: wait until a knight leaves spawn, then use the arrow that already drops 16. Take the heart container. The Done line needs at least 10 of 12 RNG offsets. Claim AP-104 and mark it done only when that check passes. If the knights are still alive, block AP-104 again with the new policy's frame, HP, and offset count.\n";
eastern += "You may edit snes/alttp/dungeon/bosses/armos_knights.py, snes/alttp/docs/routes/eastern_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-103/. Do not edit moldorm.py.\n";
jobs.push(#{ prompt: eastern, label: "mgr-armos", capability_mode: "all", output_schema: lane_schema });

let wave = parallel(jobs);
let n = 0;
if wave != () {
    for slot in wave {
        if slot != () && slot.success {
            n += 1;
        }
    }
}
log("lanes returned " + n.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-104, AP-111, AP-120, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Fourth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
