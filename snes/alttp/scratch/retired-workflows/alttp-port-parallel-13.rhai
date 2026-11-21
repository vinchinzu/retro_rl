let meta = #{
    name: "alttp-port-parallel-13",
    description: "Sword Kholdstare west of the spawn shard, and leave the library with the book already in hand",
    when_to_use: "After alttp-port-parallel-12. UP and RIGHT die on the shard. The tablet walk used a pin with book 0.",
    phases: [
        #{ title: "Lanes", detail: "Kholdstare shard, library exit" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight. Do not edit moldorm.py.\n";
ice += "Closed exits, all with the eight fire shots kept and the heart not taken: DOWN frame 2001, LEFT frame 1410, UP from (7482,6809) stopped (7485,6760) and died frame 1440 to three ice-16 hits, RIGHT stepped into the shard and died frame 1316, UP-then-RIGHT stuck (7543,6759) and died frame 1472. Do not repeat those five paths.\n";
ice += "The new fact is the spawn shard: 15 px east at (7497,6809), z=215. The south-window master sword is the hit that deals 4. Swing there, then step to an x at least 16 px west of 7497 before the shard falls. Score offset 0 only. If sword_hits stays 0 or the death is still ice 16, note the pose and leave the claim [~]. Do not score offsets 1-11 and do not block AP-168. Wall rows 27-38 stay unwalked.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py and recordings under snes/alttp/recordings/probe_AP-168/.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet. Do not claim AP-112 or AP-114.\n";
desert += "book_play.state has the play book in room 0x0107, and its exit iris hangs. The walk that was tried used a different pin: outdoor road (3216,2344), screen 0x26, book 0. Eastern exits from there only go south. West from screens 0x2C and 0x34 wraps link_x to 65533. Screen 0x2D reaches 0x25, and the west step is a 41-cell ledge at (2528,2216) whose exits return to 0x25. The plaque nearest (296,3259) was not stood. Do not repeat that walk and do not boot the book-0 pin.\n";
desert += "Leave room 0x0107 from book_play.state. If the iris hangs, use the door transition that actually changes the room instead of waiting on the iris. The tablet read has to happen in play with the book byte already 1. Mark AP-113 done only when that read passes the Done line. If you are still indoors or still on screen 0x25, note the pose and leave the claim [~]. Do not block AP-113.\n";
desert += "You may edit snes/alttp/docs/routes/desert_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-113/. Do not edit kholdstare.py.\n";
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-113, AP-120, AP-165, and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Thirteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
