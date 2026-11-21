let meta = #{
    name: "alttp-port-parallel-8",
    description: "Patch Moldorm's five misses without dropping the 7 of 12, find a new library door, and probe Kholdstare",
    when_to_use: "After alttp-port-parallel-7. Moldorm is restored to 7 of 12. The library dashes stayed outdoors.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm patch, library door, Kholdstare" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm, Whatif_Eastern_armos, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight. AP-120 is [~] moldorm-fight.\n";
hera += "snes/alttp/bosses/moldorm.py is the restored 7/12 fight. eval_offsets on Whatif_Hera_moldorm hearts on offsets 0, 1, 2, 3, 4, 7, 10 at frames 982, 821, 859, 978, 852, 929, 830. The same five still miss: 5 frame 1043 one Moldorm hit, 6 frame 677 floor change, 8 frame 1384 one hit, 9 cap 7500, 11 frame 779 one hit. A floor-clamp and sidestep edit scored 4/12 and was reverted. Do not repeat that edit.\n";
hera += "Before you change the file, copy it to snes/alttp/recordings/probe_AP-120/moldorm_7of12.py. Score offset 0 after every edit. If offset 0 misses the heart or takes longer than 1200 frames, copy the backup back over moldorm.py and stop. Do not score offsets 1-11 on a regressed fight.\n";
hera += "Change only the path that runs on offsets 5, 6, 8, 9, and 11. Mark AP-120 done only at 10 of 12 hearts. If you stay under 10, note the pass list and leave the claim [~]. Do not block AP-120.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py and recordings under snes/alttp/recordings/probe_AP-120/. Do not edit lanmolas.py, primitives.py, or walk_physics.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let book = common();
book += "You are the Book of Mudora lane. Claim name is exactly book-mudora. AP-090 is [~] book-mudora.\n";
book += "Closed dashes on pin recordings/probe_AP-089/boots.state, follower 0x05, book $F34E stayed 0. Screen 0x2B LEFT from (1720,2992) bonks at (1650,2992), front attr 0x00, no west exit. Screen 0x1B LEFT from (1696,1848) stops in text at (1573,1848), front attr 0x40, still screen 0x1B. Do not dash those two lines again.\n";
book += "Read z3-json-data for the library screen and the bookshelf. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. One new approach from the boots pin. The book byte has to rise in play. If the data names no screen besides 0x2B and 0x1B, leave the claim [~] and note that. Do not block AP-090. If $F34E rises, mark AP-090 done with the pin, the screen, and the ledger frame.\n";
book += "You may edit recordings under snes/alttp/recordings/probe_AP-090/. Do not edit boss modules, primitives.py, or walk_physics.py.\n";
jobs.push(#{ prompt: book, label: "mgr-book", capability_mode: "all", output_schema: lane_schema });

let ice = common();
ice += "You are the Kholdstare probe lane. AP-165 stays blocked. Do not press A on the room 0x1F chest, and do not edit walk_physics.py.\n";
ice += "Read snes/alttp/dungeon/route_data/ice.yaml for the Kholdstare room id. Do not invent one. Copy the warp in snes/alttp/recordings/probe_AP-103/load_armos.py into your own script: warp submodule 0x15, previous room kept, $A0 set to the Kholdstare room, never write $F3CC. Source pin is an indoor Ice pin that already has the fire rod in play if one exists. Save Whatif_Ice_kholdstare only when pin_probe shows that room and the boss sprite is on screen.\n";
ice += "AP-167 is the probe. Do it only when the sprite is on screen. Leave it open and note the pin if the room does not load. Do not block it for an unwalked path. Do not start AP-168 unless the probe's Done line is met and it asks for the fight file.\n";
ice += "You may edit snes/alttp/docs/routes/ice_ROUTE.md, snes/alttp/bosses/kholdstare.py only if AP-168 is actually reached, and recordings under snes/alttp/recordings/probe_AP-167/. Do not edit moldorm.py.\n";
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-090, AP-112, AP-120, AP-165, and AP-167. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Eighth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
