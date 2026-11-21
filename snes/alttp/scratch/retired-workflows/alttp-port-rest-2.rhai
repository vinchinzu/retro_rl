let meta = #{
    name: "alttp-port-rest-2",
    description: "Somaria, the pipe ride, and the three boss probes those dungeons have not tried",
    when_to_use: "After alttp-port-rest. Do not retry the pit, the shutter, or the GT big-key walk.",
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
    p += "\nRe-read each heading. Skip [x], [!], and [~] owned by someone else. A step is done only when its Done line passes. Two honest tries, then block that step. Do not weaken a Done line. A poke does not pass a line that requires the item to rise in play. A boss probe names the pin and says isolated poke. A boss policy needs at least 10 of 12 RNG offsets. Hint lines are not authority. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download refs.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Re-read a file immediately before you edit it. Geometry stays in maps json. Do not copy approach xy into Python. Soft max about 1000 lines. primitives.py is past that cap. Add a small function there. Do not extract a sibling module.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
    return p;
}

phase("Lanes");
let jobs = [
    #{
        prompt: lane_prompt(
            "primitives",
            "AP-174, then AP-181",
            "You are the only agent who may edit primitives.py. Add somaria_block, then ride_platform. fire_rod, hammer, and hookshot already exist. Use them as the pattern. ride_platform places a somaria block on tile attribute 0xB6, stands on it, and presses at a junction. Prove each function on a live tile. Scan_mire_0x98 is room 0x98. Scan_turtle_rock_0xd6 is room 0xD6.",
            "Mire room 0x98 pit columns 36-41 rows 39-42. Zol HP stayed 4. Hookshot did not cross. Turtle Rock UP from (3448,7128) stuck at (3448,7016) and the north path was empty. Do not retry those walks.",
            "snes/alttp/primitives.py, the primitive test that covers hookshot, snes/alttp/docs/routes/mire_ROUTE.md, snes/alttp/docs/routes/turtle_rock_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-174/"
        ),
        label: "mgr-AP-174",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "tower",
            "AP-127, then AP-128 only if the probe shows the reflect window",
            "CastleTowerEntry is room 0xE0. Read Agahnim's room id from z3 or the route yaml. A boss-room load is allowed for the probe, the same way Helmasaur used Scan_pod_boss. Do not invent the room. Do not edit primitives.py.",
            "AP-125. The east door nudge died at (200,7299). The 0x6A fight left both soldiers at 16 HP. KeyA1 stayed shut. Do not retry that door.",
            "snes/alttp/docs/routes/castle_tower_ROUTE.md, snes/alttp/dungeon/route_data/castle_tower.yaml, snes/alttp/dungeon/bosses/agahnim.py, recordings under snes/alttp/recordings/probe_AP-127/"
        ),
        label: "mgr-AP-127",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "mire",
            "AP-175 only",
            "Read the Vitreous room id from mire yaml or z3. A boss-room load is allowed for the probe. Do not invent the room. Do not edit primitives.py or mire_ROUTE.md. Another lane owns that route doc.",
            "AP-171 and AP-173. The lobby pit was not crossed. Zol HP stayed 4. Hookshot did not make a crossing ancilla. Do not retry that hop.",
            "snes/alttp/dungeon/bosses/vitreous.py, recordings under snes/alttp/recordings/probe_AP-175/"
        ),
        label: "mgr-AP-175",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "gt",
            "AP-191 only",
            "GanonsTowerEntry is room 0x0C, and the north stair reaches room 0x6B. Read the Agahnim 2 room id from gt yaml or z3. A boss-room load is allowed for the probe. Do not invent the room. Do not edit bosses/agahnim.py. The Tower lane owns that file. Do not edit primitives.py.",
            "AP-189. Room 0x1C was not entered. Eyegore slots did not change. The big key stayed 0. Do not retry that walk.",
            "snes/alttp/docs/routes/gt_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-191/"
        ),
        label: "mgr-AP-191",
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
log("rest-2 lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, which of AP-174, AP-175, AP-181, AP-127, and AP-191 changed from open, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Rest wave 2 ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
