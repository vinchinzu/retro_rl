let meta = #{
    name: "alttp-port-parallel-14",
    description: "Swing at Kholdstare before the shard splits, and read the Desert tablet from an overworld pin that already has the book",
    when_to_use: "After alttp-port-parallel-13. The sword landed after the shard split. The library pocket has no exit.",
    phases: [
        #{ title: "Lanes", detail: "Early swing, tablet pin" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Hera_moldorm, book_play.state, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight. Do not edit moldorm.py.\n";
ice += "Offset 0 on Whatif_Ice_kholdstare_magic: the south-window sword dealt 4 at frame 1091, pose (7482,6747) facing UP, while the spawn ball was still at z=36. The 13-frame swing finished after the split. The west step then read (7480,6742), 17 px west of 7497, with z already 0, and died at frame 1419 pose (7464,6723) to three falling-ice hits. Log recordings/probe_AP-168/offset0_west2.log. Do not repeat that post-split step, and do not repeat UP, DOWN, LEFT, or RIGHT.\n";
ice += "Start the swing early enough that it finishes while z is still above 0, and be at x<=7481 before the ball splits. Score offset 0 only. If the swing still ends after the split, note the frame and leave the claim [~]. Do not score offsets 1-11 and do not block AP-168. Eight fire shots stay. Wall rows 27-38 stay unwalked.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py and recordings under snes/alttp/recordings/probe_AP-168/.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet. Do not claim AP-112 or AP-114.\n";
desert += "book_play.state is stuck. Pose (3643,8504), book 1, quadrant (0,2). The reachable pocket is local x 40..200 and y 304..488, 196 cells, no stairs. The only south cells at y>=476 are two attribute 0x8E iris tiles. A room change south needs a non-0x8E tile, and a room change west needs x<8. Neither is reachable. Do not walk that pocket again, and do not boot the book-0 road pin at (3216,2344).\n";
desert += "The book byte already rose in play in room 0x0107 at ledger frame 377. Copy that risen value onto an overworld pin that can stand at the plaque nearest (296,3259). Record the copy as a poke. The tablet read itself has to happen in play. Mark AP-113 done only when that read passes the Done line. If the plaque still cannot be stood, note the screen and the pose and leave the claim [~]. Do not block AP-113.\n";
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
    summary: "Fourteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
