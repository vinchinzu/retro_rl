let meta = #{
    name: "alttp-port-parallel-10",
    description: "Branch Moldorm off the 8 of 12 file, dash the library shelf in room 0x3D, and finish the Kholdstare shell",
    when_to_use: "After alttp-port-parallel-9. Shell fire stops at 24 because magic hits 0.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm branch, library shelf, shell break" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm, Whatif_Ice_kholdstare, Whatif_Library, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes. A boss step needs the heart on at least 10 of 12 offsets when that is the Done line.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight. AP-120 is [~] moldorm-fight.\n";
hera += "snes/alttp/bosses/moldorm.py is the 8/12 file. Passes on Whatif_Hera_moldorm, hp_lost 0: offsets 0, 1, 2, 4, 7, 9, 10, 11 at frames 982, 820, 855, 849, 929, 850, 828, 979. Fails: 3 frame 711 hp_lost 8, 5 frame 421 hp_lost 16, 6 frame 676 hp_lost 8, 8 frame 950 hp_lost 16. A closing-head sidestep floored offset 0 at frame 706 and was reverted from moldorm_8of12.py. Do not repeat that sidestep or the 4/12 floor-clamp.\n";
hera += "Copy the current file to snes/alttp/recordings/probe_AP-120/moldorm_8of12.py before you edit, replacing that backup only if you have confirmed the current file is this 8/12 fight. The new code has to branch on the RNG offset. Offset 0's path must stay byte-for-byte the current path. Score offset 0 after the edit. If it misses the heart or takes longer than 1200 frames, restore the backup and stop. Do not score the other offsets on a regressed fight.\n";
hera += "You need two more hearts among offsets 3, 5, 6, and 8. Mark AP-120 done only at 10 of 12. If you stay under 10, note the pass list and leave the claim [~]. Do not block AP-120.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py and recordings under snes/alttp/recordings/probe_AP-120/. Do not edit kholdstare.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let book = common();
book += "You are the Book of Mudora lane. Claim name is exactly book-mudora. AP-090 is [~] book-mudora.\n";
book += "Whatif_Library is indoors room 0x003D at (6774,10120), warped from probe_AP-089/boots.state with previous room 0x0105. Sprites were none, dash-item 0x3B was absent, and $F34E stayed 0. The bookshelf is a tile, not that sprite. Do not dash screen 0x2B or screen 0x1B again, and do not warp a second room id.\n";
book += "Scan room 0x3D for the shelf tile and dash into it from this pin. Mark AP-090 done only when $F34E rises in play, with the ledger frame. If a full scan shows no dashable shelf, leave the claim [~] and note the tile ids you read. Do not block AP-090.\n";
book += "You may edit recordings under snes/alttp/recordings/probe_AP-090/. Do not edit boss modules or primitives.py.\n";
jobs.push(#{ prompt: book, label: "mgr-book", capability_mode: "all", output_schema: lane_schema });

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight. Do not touch AP-165.\n";
ice += "On Whatif_Ice_kholdstare, five fire shots drop the shell 64, 56, 48, 40, 32, 24 while magic goes 80 to 0. Sword hits stayed 0. The eye stayed ai 3 hp 64. End pose (7507,6748). Each shot costs 16 magic and 8 shell HP, so magic 80 cannot finish a 64 HP shell. Wall rows 27-38 stay unwalked.\n";
ice += "Copy the pin to Whatif_Ice_kholdstare_magic and set magic to 128. Record that write as a poke. Do not poke shell HP. Fire until the shell reaches 0, then sword the eye, off the falling-ice column. Mark AP-168 done only when the heart is taken on at least 10 of 12 offsets. If the shell reaches 0 and the eye never becomes vulnerable, note the eye HP and AI and leave the claim [~]. Do not block AP-168.\n";
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
    summary: "Tenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
