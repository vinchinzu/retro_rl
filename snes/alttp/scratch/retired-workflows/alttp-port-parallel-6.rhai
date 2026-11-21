let meta = #{
    name: "alttp-port-parallel-6",
    description: "Moldorm's five misses, Lanmolas on the cardinal gap, and Sahasrahla's boots receipt",
    when_to_use: "After alttp-port-parallel-5. AP-106 is done. AP-120 is 7 of 12.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm, Lanmolas, boots" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Eastern_armos, Whatif_Desert_lanmolas, Whatif_Hera_moldorm, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight. AP-120 is [~] moldorm-fight. Do not touch AP-118 or AP-119.\n";
hera += "Closed result on Whatif_Hera_moldorm: hearts on offsets 0, 1, 2, 3, 4, 7, 10 (frames 982, 821, 859, 978, 852, 929, 830). Failures: offset 5 frame 1043 hp_lost 8 moldorm:1; offset 6 frame 677 fell to room 0x17; offset 8 frame 1384 hp_lost 8 moldorm:1; offset 9 cap 7500 hp_lost 0; offset 11 frame 779 hp_lost 8 moldorm:1. Do not rerun the seven passing offsets as the policy to change, and do not repeat the walked rel=(-4,0) whiff.\n";
hera += "Read snes/alttp/bosses/moldorm.py. Keep the passing offsets. Change the fight so offsets 5, 6, 8, 9, and 11 also take the heart: do not take the contact hit, and do not walk off the floor into room 0x17. Score all 12. Mark AP-120 done only at 10 of 12 hearts. If you are still under 10, note the new pass list and leave the claim [~] under moldorm-fight. Do not block AP-120.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py, snes/alttp/docs/routes/hera_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-120/. Do not edit lanmolas.py or primitives.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Lanmolas fight lane. AP-111 is done. AP-112 is [!]. Whatif_Desert_lanmolas is room 0x33 with three sprite 0x54.\n";
desert += "Closed fight: offset 0 died at frame 1653, hp 24 to 0, hits boulder:1,lanmolas:2, strikes 5. Sword commits land at distance 21-25 while $0F70 is 1-11. Indoor 0xC2 travels on four diagonals; $0D50 is X and $0D40 is Y, pixel speed signed/16. Cardinals are the gap. Do not stand on a diagonal again.\n";
desert += "Read snes/alttp/bosses/lanmolas.py. New policy: stand on a cardinal gap and swing only in that window. Claim AP-112. Mark it done only when the heart is taken on at least 10 of 12 offsets. If offset 0 dies to a boulder again, block AP-112 with the new pose, frame, and hit list.\n";
desert += "You may edit snes/alttp/bosses/lanmolas.py, snes/alttp/docs/routes/desert_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-112/. Do not edit moldorm.py or primitives.py.\n";
jobs.push(#{ prompt: desert, label: "mgr-lanmolas", capability_mode: "all", output_schema: lane_schema });

let boots = common();
boots += "You are the Pegasus Boots lane. AP-084 and AP-106 are done. Claim AP-089. Do not claim AP-086, AP-090, or AP-109.\n";
boots += "Closed fact: the only store of ability bit 0x04 is STA.l at $09:8658 during the boots receipt, and that receipt skips when $F355 is already 1. Do not search for another writer. Do not load DesertPalaceEntry or Whatif_DesertPalaceEntry_abilityFC. Do not press A at sprite 0x3B.\n";
boots += "AP-106 already raised pendant $F374 from 0x00 to 0x04 in play on Whatif_Eastern_armos. Start from a pin where that pendant bit is set and boots $F355 is 0. If the Armos pin already has boots 1, copy the pendant bit onto a pin whose boots byte is 0 and record the copy as a poke. Do not poke boots or the dash bit.\n";
boots += "Sahasrahla's hut is maps/screen_1e_sahasrahla.json. Use that door. Do not repeat the AP-086 library walks. Talk. The boots byte has to rise from the receipt. Then measure dash charge frames, speed, and the stop on a wall, and add a dash primitive in snes/alttp/primitives.py. Mark AP-089 done only when the Done line passes: boots rose in play, and the dash constants are measured. If you cannot stand in the hut, leave AP-089 open and note the pose.\n";
boots += "You may edit snes/alttp/primitives.py, snes/alttp/docs/routes files that AP-089 names, and recordings under snes/alttp/recordings/probe_AP-089/. Do not edit boss modules or walk_physics.py.\n";
jobs.push(#{ prompt: boots, label: "mgr-boots", capability_mode: "all", output_schema: lane_schema });

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-089, AP-112, AP-120, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Sixth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
