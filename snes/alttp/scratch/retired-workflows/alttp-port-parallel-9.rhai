let meta = #{
    name: "alttp-port-parallel-9",
    description: "Lift Moldorm from 8 of 12, dash the book inside library room 61, and fight Kholdstare",
    when_to_use: "After alttp-port-parallel-8. AP-167 is done. Moldorm is 8 of 12.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm, library room, Kholdstare" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm, Whatif_Ice_kholdstare, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes. Boss step 6 needs the heart on at least 10 of 12 RNG offsets when that is the Done line.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight. AP-120 is [~] moldorm-fight.\n";
hera += "The current snes/alttp/bosses/moldorm.py scores 8 of 12 on Whatif_Hera_moldorm. Passes, hp_lost 0: offset 0 frame 982, 1 frame 820, 2 frame 855, 4 frame 849, 7 frame 929, 9 frame 850, 10 frame 828, 11 frame 979. Fails: 3 frame 711 hp_lost 8 moldorm:1, 5 frame 421 hp_lost 16 moldorm:2, 6 frame 676 hp_lost 8 moldorm:1, 8 frame 950 hp_lost 16 moldorm:2. Offset 9 swings a tail that was paused 24 frames. Offset 11 keeps the off-camera deflection wake after HP hits 0. Log recordings/probe_AP-120/eval_wake.log.\n";
hera += "This 8/12 file traded away offset 3, which the older 7/12 file passed. Keep offsets 9 and 11. Copy the current file to snes/alttp/recordings/probe_AP-120/moldorm_8of12.py before you edit. After every edit, score offset 0 only. If it misses the heart or takes longer than 1200 frames, copy moldorm_8of12.py back and stop. Do not restore the older 7/12 backup over this file.\n";
hera += "Change only offsets 3, 5, 6, and 8. You need two more hearts. Mark AP-120 done only at 10 of 12. If you stay under 10, note the pass list and leave the claim [~]. Do not block AP-120. Do not repeat the floor-clamp sidestep that scored 4/12.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py and recordings under snes/alttp/recordings/probe_AP-120/. Do not edit kholdstare.py or primitives.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let book = common();
book += "You are the Book of Mudora lane. Claim name is exactly book-mudora. AP-090 is [~] book-mudora.\n";
book += "Closed: z3-json-data names the Library as Light World door id 26 and cave room id 61, item BookOfMudora at nodeAddress 0x180012. That connection has no overworld screen. The dashes on screen 0x2B from (1720,2992) and screen 0x1B from (1696,1848) stayed outdoors and left $F34E at 0. Do not dash those lines again.\n";
book += "Read the json and use the room id it stores. Do not invent a second id. Copy the warp in snes/alttp/recordings/probe_AP-103/load_armos.py into your own script: submodule 0x15, previous room kept, $A0 set to that room, never write $F3CC. Source pin is recordings/probe_AP-089/boots.state, which already has the boots receipt. Save a Whatif pin only when you are indoors in that room.\n";
book += "Dash the bookshelf in play. Mark AP-090 done only when $F34E rises, with the pin name and the ledger frame. If the room will not load or the byte stays 0, leave the claim [~] and note the room and the byte. Do not block AP-090.\n";
book += "You may edit recordings under snes/alttp/recordings/probe_AP-090/. Do not edit boss modules or primitives.py.\n";
jobs.push(#{ prompt: book, label: "mgr-book", capability_mode: "all", output_schema: lane_schema });

let ice = common();
ice += "You are the Kholdstare fight lane. AP-167 is done. Claim AP-168. Do not touch AP-165 or AP-166.\n";
ice += "Whatif_Ice_kholdstare is room 0xDE. Shell sprite 0xA3 HP 64 AI 0, Kholdstare 0xA2 HP 64 AI 3 at (7544,6744). Falling ice 0xA4 at frame 125, shards at frame 191, and Link HP 64 to 0 at frames 192, 320, 448, and 576 on an idle stand. On the unsaved north floor, fire dropped magic 80 to 64 and the shell 64 to 56 at frame 11. A master-sword hit 16 px south left the shell at 64. Wall rows 27-38 were not walked. Do not walk them.\n";
ice += "Read the AP-168 Done line. Write snes/alttp/bosses/kholdstare.py: fire the shell, then the sword on the pieces, and do not stand in the falling ice. Mark AP-168 done only when that Done line passes. If the shell never drops below 56, note the HP and leave AP-168 open. Do not block it for the idle ice deaths already recorded.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py, snes/alttp/docs/routes/ice_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-168/. Do not edit moldorm.py or walk_physics.py.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-090, AP-112, AP-120, AP-165, and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Ninth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
