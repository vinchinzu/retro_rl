let meta = #{
    name: "alttp-port-rest",
    description: "Four dungeon lanes at once: Tower, Mire, Turtle Rock, and Ganon's Tower",
    when_to_use: "While Blind is running. These four have no start pin. Do not touch Thieves' Town files.",
    phases: [
        #{ title: "Dungeons", detail: "four managers in parallel" },
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

fn lane_prompt(name, lo, hi, rule, files) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are manager ";
    p += name;
    p += ". Tasks ";
    p += lo;
    p += " through ";
    p += hi;
    p += ". Another agent is already running the Blind fight. Do not edit snes/alttp/dungeon/bosses/blind.py, snes/alttp/docs/routes/thieves_ROUTE.md, or anything under recordings/probe_AP-160/. Do not kill other Python processes.\n";
    p += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += "Do not edit primitives.py, make_entry_pin.py, whatif_pin.py, ram.py, nav.py, or another dungeon's files. If a step needs a new shared primitive, set waiting_on to that id, write the needed change in your route doc, and continue to a later step that does not need it.\n";
    p += rule;
    p += "\nRe-read each heading. Skip [x], [!], and [~] owned by someone else. A step is done only when its Done line passes. Two honest tries, then block that step. After two steps you yourself blocked, stop. Do not weaken a Done line. A poke does not pass a line that requires the item to rise in play. Boss step 6 needs at least 10 of 12 RNG offsets. A pin is isolated, not natural, not continuous, and not STATUS. Hint lines are not authority. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download refs.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Re-read a file immediately before you edit it. Geometry stays in maps json. Do not copy approach xy into Python. Soft max about 1000 lines. Do not extract a sibling module.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator in your process. Pytest uses QT_QPA_PLATFORM=offscreen. Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
    return p;
}

let table = "The US dungeon entrance table is named in snes/alttp/scratch/probe_dp_entry.py: ROM $02C813, index 0x09 points at room 0x0084. Overworld door index 0x08 is a different table. Read that file before you map a SNES address to a ROM file offset. Confirm your mapping by reading index 0x09 and getting room 0x0084. If that check fails, do not try other offsets and do not boot. Find your dungeon's entrance room from your yaml or from z3, then find the table index whose room word matches. trigger_entrance writes $010E and module 0x06 or 0x0F, the same way make_entry_pin.py does. Save a pin only if pin_probe shows indoors, module 0x07, and that room. A castle overlay or dungeon id 0x02 is not a save. Do not poke $A0. Do not invent an index. Your probe script lives under your own recordings directory, not in make_entry_pin.py.";

phase("Dungeons");
let jobs = [
    #{
        prompt: lane_prompt(
            "tower",
            "AP-123",
            "AP-130",
            table + " Pin name CastleTowerEntry. Kit on a real load may include the lamp. Do not cut the barrier by writing the sword.",
            "snes/alttp/docs/routes/castle_tower_ROUTE.md, snes/alttp/dungeon/route_data/castle_tower.yaml, snes/alttp/dungeon/bosses/agahnim.py, maps for rooms you enter, recordings under snes/alttp/recordings/probe_AP-123/"
        ),
        label: "mgr-AP-123",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "mire",
            "AP-171",
            "AP-178",
            table + " Pin name MiseryMireEntry. Opening the entrance with Ether is AP-177, not the pin.",
            "snes/alttp/docs/routes/mire_ROUTE.md, snes/alttp/dungeon/route_data/mire.yaml, snes/alttp/dungeon/bosses/vitreous.py, maps for rooms you enter, recordings under snes/alttp/recordings/probe_AP-171/"
        ),
        label: "mgr-AP-171",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "turtle",
            "AP-179",
            "AP-186",
            table + " Pin name TurtleRockEntry. Quake in play is AP-185, not the pin.",
            "snes/alttp/docs/routes/turtle_rock_ROUTE.md, snes/alttp/dungeon/route_data/turtle_rock.yaml, snes/alttp/dungeon/bosses/trinexx.py, maps for rooms you enter, recordings under snes/alttp/recordings/probe_AP-179/"
        ),
        label: "mgr-AP-179",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "gt",
            "AP-187",
            "AP-194",
            table + " Pin name GanonsTowerEntry. Do not edit bosses/agahnim.py. The Tower lane owns that file. If you reach the Agahnim step, set waiting_on AP-192 and stop.",
            "snes/alttp/docs/routes/gt_ROUTE.md, snes/alttp/dungeon/route_data/gt.yaml, maps for rooms you enter, recordings under snes/alttp/recordings/probe_AP-187/"
        ),
        label: "mgr-AP-187",
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
log("rest lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, which of AP-123, AP-171, AP-179, and AP-187 changed, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Rest wave ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
