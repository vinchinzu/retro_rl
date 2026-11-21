let meta = #{
    name: "alttp-port-parallel-7",
    description: "Restore the Moldorm 7 of 12 fight, dodge Lanmolas boulders, and dash the library book",
    when_to_use: "After alttp-port-parallel-6. AP-089 is done. The latest Moldorm eval scored 0 of 12.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm restore, Lanmolas dodge, book" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm, Whatif_Desert_lanmolas, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight. AP-120 is [~] moldorm-fight.\n";
hera += "The current snes/alttp/bosses/moldorm.py is a regression. Its latest eval_offsets on Whatif_Hera_moldorm scored 0 of 12 hearts: offset 0 frame 2213 hp_lost 24 hits moldorm:3, offsets 1-11 cap 7500 with no hits. That file is untracked, so git has no older copy.\n";
hera += "The last good score is only in the AP-120 learned note: hearts on offsets 0, 1, 2, 3, 4, 7, 10 at frames 982, 821, 859, 978, 852, 929, 830. Failures of THAT policy were offset 5 frame 1043 one Moldorm hit, offset 6 frame 677 room 0x17, offset 8 frame 1384 one hit, offset 9 cap 7500, offset 11 frame 779 one hit.\n";
hera += "Search recordings and scratch for the fight code that produced those ~800 frame hearts. Restore that behavior. Prove it by scoring offset 0 only, and stop if offset 0 does not take the heart. Do not run offsets 1-11 until offset 0 hearts again. Then fix only offsets 5, 6, 8, 9, and 11. Mark AP-120 done only at 10 of 12 hearts. If offset 0 still fails, note the frame and leave the claim [~]. Do not block AP-120.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py, snes/alttp/docs/routes/hera_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-120/. Do not edit lanmolas.py or primitives.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Lanmolas fight lane. AP-112 is [!]. Do not repeat the cardinal-gap policy.\n";
desert += "Closed death on Whatif_Desert_lanmolas: offset 0 died at frame 2894 pose (1683,1918) facing LEFT, hp 24 to 0, hits boulder:2,lanmolas:2. Hurt frames: 1028 UP diag=1, 1134 UP diag=1, 2164 RIGHT rock dx=-7 dy=9, 2894 LEFT with no rock. Five strikes on cardinals did not save the attempt. Standing on a cardinal is not a dodge.\n";
desert += "Read snes/alttp/bosses/lanmolas.py. New policy: when a boulder has both dx and dy nonzero, step off that ray before contact. A worm hit still counts, so do not trade a boulder dodge for a lanmolas hit. Claim AP-112. Mark it done only at 10 of 12 hearts. If offset 0 dies again, block AP-112 with the new pose, the boulder dx/dy, and the hit list.\n";
desert += "You may edit snes/alttp/bosses/lanmolas.py and recordings under snes/alttp/recordings/probe_AP-112/. Do not edit moldorm.py.\n";
jobs.push(#{ prompt: desert, label: "mgr-lanmolas", capability_mode: "all", output_schema: lane_schema });

let book = common();
book += "You are the Book of Mudora lane. AP-089 is done. Claim AP-090. Do not claim AP-086.\n";
book += "Dash is measured: charge 30 frames, then state 17 at 4 px/frame, stop on attribute 0x27. The boots receipt in hut room 0x05 set $F355 0 to 1 and ability $F379 0xF8 to 0xFC in play. Use a pin that already has that receipt. Do not poke the book byte $F34E.\n";
book += "AP-086 on foot never entered the library: screen 0x2B has no west exit, and the screen 0x1B west strip y=1816-1848 does not change area. Do not repeat those two walks without a dash. One dash along each of those approaches is a new fact. The book has to rise in play from the bookshelf dash, on an isolated leg.\n";
book += "If both dash approaches stay outside the library, leave AP-090 open and note the poses. Do not block it for the old on-foot result. If the book byte rises, mark AP-090 done with the pin name and the ledger frame.\n";
book += "You may edit a route note the task names and recordings under snes/alttp/recordings/probe_AP-090/. Do not edit boss modules, primitives.py, or walk_physics.py.\n";
jobs.push(#{ prompt: book, label: "mgr-book", capability_mode: "all", output_schema: lane_schema });

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-090, AP-112, AP-120, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Seventh wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
