let meta = #{
    name: "alttp-port-rest-3",
    description: "Vitreous probe, Trinexx probe, Tower upper floors, and the GT red-mail room",
    when_to_use: "After alttp-port-rest-2. AP-175 is a stale claim. Do not retry the blocked fights or the pipe.",
    phases: [
        #{ title: "Lanes", detail: "four file-disjoint managers" },
        #{ title: "Report", detail: "ticks and credits tape" },
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
    "required": ["summary", "open_count", "done_count", "blocked_count", "end_to_end"],
    "properties": #{
        "summary": #{ "type": "string" },
        "open_count": #{ "type": "integer" },
        "done_count": #{ "type": "integer" },
        "blocked_count": #{ "type": "integer" },
        "end_to_end": #{ "type": "boolean" },
    },
};

fn lane_prompt(name, tasks, rule, closed, files) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are manager ";
    p += name;
    p += ". Tasks: ";
    p += tasks;
    p += ".\n";
    p += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += rule;
    p += "\nClosed, do not repeat: ";
    p += closed;
    p += "\nRe-read each heading. Skip [x] and [!]. A step is done only when its Done line passes. Two honest tries, then block that step. Do not weaken a Done line. A poke does not pass a line that requires the item to rise in play. A boss probe names the pin and says isolated poke. Read the boss room id from that dungeon's yaml or z3 data. Do not invent it. A boss-room load is allowed for a probe, the same way Helmasaur used Scan_pod_boss. It is not an entry pin. Hint lines are not authority. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download refs.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Do not edit primitives.py, make_entry_pin.py, or bosses/agahnim.py. Re-read a file immediately before you edit it. Geometry stays in maps json. Do not copy approach xy into Python. Soft max about 1000 lines. Do not extract a sibling module.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
    return p;
}

phase("Lanes");
let jobs = [
    #{
        prompt: lane_prompt(
            "mire",
            "AP-175, then AP-176 only if the probe shows what hurts Vitreous",
            "Claim AP-175 with the agent name mire. The heading is already [~] mire with no evidence. That same name may claim it again. Any other name is refused.",
            "AP-171, AP-173, and AP-174. The lobby pit was not crossed. Somaria ancilla 0x2C left $0646 at 0. Room 0x98 has no block tile. Do not retry that lobby.",
            "snes/alttp/dungeon/bosses/vitreous.py, snes/alttp/docs/routes/mire_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-175/"
        ),
        label: "mgr-AP-175",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "turtle",
            "AP-183 only",
            "Scan_turtle_rock_0xd6 is room 0xD6. Use it only as the source state for a boss-room load. Do not ride the rail.",
            "AP-179 and AP-181. UP stuck at (3448,7016). ride_platform spawned sprite 0xED and $02F5 stayed 0. Do not retry that ride.",
            "snes/alttp/dungeon/bosses/trinexx.py, snes/alttp/docs/routes/turtle_rock_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-183/"
        ),
        label: "mgr-AP-183",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "tower",
            "AP-126 only",
            "CastleTowerEntry is room 0xE0. Agahnim's own room was already probed. Do not refight him.",
            "AP-125 and AP-128. The east door died at (200,7299). KeyA1 stayed shut. The reflect policy died with the boss still at 80 HP. Do not retry that door or that fight.",
            "snes/alttp/docs/routes/castle_tower_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-126/"
        ),
        label: "mgr-AP-126",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "gt",
            "AP-190 only",
            "GanonsTowerEntry is room 0x0C and the north stair reaches 0x6B. Read the red-mail room from gt yaml or z3. If the task's Done line requires walking the blocked big-key door, leave AP-190 open and do not boot a long walk.",
            "AP-189 and AP-191. Room 0x1C was not entered. Agahnim 2 rooms 0x0D and 0x20 showed one body and no clone. Do not retry either.",
            "snes/alttp/docs/routes/gt_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-190/"
        ),
        label: "mgr-AP-190",
        capability_mode: "all",
        output_schema: lane_schema,
    },
];
let wave = parallel(jobs);
let wave_done = 0;
if wave != () {
    for slot in wave {
        if slot != () && slot.success {
            wave_done += 1;
        }
    }
}
log("rest-3 lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, the mark on AP-126, AP-175, AP-183, and AP-190, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Rest wave 3 ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
