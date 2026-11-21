let meta = #{
    name: "alttp-port-parallel-12",
    description: "Leave the Kholdstare window up or right, and read the Desert tablet with the book",
    when_to_use: "After alttp-port-parallel-11. AP-090 is done. DOWN and LEFT both die to the ice.",
    phases: [
        #{ title: "Lanes", detail: "Kholdstare exits, Desert tablet" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Ice_kholdstare_magic, Whatif_Library, Whatif_Hera_moldorm, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight. Do not edit snes/alttp/bosses/moldorm.py.\n";
ice += "Keep the eight fire shots on Whatif_Ice_kholdstare_magic. The shell goes 64 to 0 and three eyes spawn. A south-window sword deals 4. Two exits are closed: DOWN died at frame 2001, HP 64 to 32 with an eye at chebyshev 32 on (7525,6848), then ice dropped 32 to 16, and the sword whiffed. LEFT died at frame 1410, HP 64 to 48 from ice at chebyshev 14, then 48 to 16 with an eye at chebyshev 4, sword_hits 0. Do not leave DOWN or LEFT again.\n";
ice += "Try UP, then RIGHT, only if UP dies. Hit for the measured 4 and be off the ice column and outside eye contact before the next shard. Mark AP-168 done only when the heart is taken on at least 10 of 12 offsets. If both new exits die to ice 16 and eye contact 32, note the poses and leave the claim [~]. Do not block AP-168. Wall rows 27-38 stay unwalked.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py and recordings under snes/alttp/recordings/probe_AP-168/. Do not edit walk_physics.py.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert tablet lane. AP-106, AP-089, and AP-090 are done. Claim AP-113. Do not claim AP-112 or AP-114.\n";
desert += "The book rose in play in room 0x0107 at ledger frame 377, book byte 0 to 1, source=play. Find that pin under snes/alttp/recordings/probe_AP-090/. Use it. Do not poke the book byte if the pin already has it.\n";
desert += "AP-113 is: read the Desert entrance tablet with the book. Read the task and snes/alttp/dungeon/route_data/desert.yaml for the tablet before you boot. Do not walk the closed Desert routes: room 0x73 shutter, screen 0x30 rock at (704,3168), entrance 0x0C stair, or the Lanmolas fight.\n";
desert += "Mark AP-113 done only when the tablet read happens in play and the Done line passes. If you cannot stand at the tablet, leave AP-113 open and note the pose. Do not block it for those closed routes.\n";
desert += "You may edit snes/alttp/docs/routes/desert_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-113/. Do not edit kholdstare.py or moldorm.py.\n";
jobs.push(#{ prompt: desert, label: "mgr-tablet", capability_mode: "all", output_schema: lane_schema });

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-090, AP-113, AP-120, AP-165, and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Twelfth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
