let meta = #{
    name: "alttp-port-parallel-15",
    description: "After the Kholdstare sword, stand off the ice that killed offset 0, and step onto the Desert plaque cells",
    when_to_use: "After alttp-port-parallel-14. The early swing landed. The south edge of the plaque did not set the flag.",
    phases: [
        #{ title: "Lanes", detail: "Post-hit tile, plaque cells" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md or moldorm.py.\n";
    p += "A pin is isolated development. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight.\n";
ice += "Offset 0 already has the early sword. Log recordings/probe_AP-168/offset0_early3.log. clear_x at frame 1032 is (7480,6809) with z=213. The sword at frame 1084 deals 4, pose (7480,6750) facing UP, and the ball is still up at the start of that swing. Death is frame 1472 at (7464,6806), three falling-ice hits. Do not retune the swing. Do not walk through (7464,6806).\n";
ice += "After that hit, stand on a tile the three ice hits do not cover. Score offset 0 only. If the heart is taken, then score all 12 and mark AP-168 done only at 10 of 12. If offset 0 still dies to falling ice, note the tile and leave the claim [~]. Do not block AP-168. Wall rows 27-38 stay unwalked.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py and recordings under snes/alttp/recordings/probe_AP-168/.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Desert tablet lane. Claim name is exactly desert-tablet. AP-113 is [~] desert-tablet.\n";
desert += "Screen 0x30 pose (304,3472), book 1, control mode, $02ED stayed 0. The 0x46 cells are (296,3464), (304,3464), (296,3472), and (304,3472). The lip is attribute 0x02 at y=3456. North from the pad (296,3259) stops in the sand at (296,3432). Do not hold north from that pad, and do not walk the library pocket.\n";
desert += "From (304,3472), step onto (304,3464) and (296,3464). Read $02ED on each. Mark AP-113 done only if the tablet flag rises in play and the Done line passes. If both cells leave $02ED at 0, note the poses and leave the claim [~]. Do not block AP-113.\n";
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-113 and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Fifteenth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
