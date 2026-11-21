let meta = #{
    name: "alttp-port-finish-2",
    description: "Fire Rod, Swamp big key with the hookshot, and a Blind probe",
    when_to_use: "After alttp-port-finish. Still not end to end. Do not retry the new blocks.",
    phases: [
        #{ title: "Interior", detail: "three unattempted steps" },
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
    p += "\nRe-read each heading. Skip [x], [!], and [~] owned by someone else. A step is done only when its Done line passes. Two honest tries, then block that step. Do not weaken a Done line. A poke does not pass a line that requires the item to rise in play. Boss probes name the pin and say isolated poke. A boss policy needs at least 10 of 12 RNG offsets. Hint lines are not authority. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download refs.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Re-read a file immediately before you edit it. Geometry stays in maps json. Do not copy approach xy into Python. Soft max about 1000 lines. Do not extract a sibling module.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen.\n";
    p += "Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
    return p;
}

phase("Interior");
let jobs = [
    #{
        prompt: lane_prompt(
            "skull",
            "AP-150, then AP-151 only if you can quote the Mothula room id",
            "SkullWoodsEntry is entrance 0x2A, room 0x58, module 0x07, dungeon 0x10. Entrance 0x29 loaded room 0x57 and is not the lobby. Add fire_rod in primitives.py next to hookshot. You are the only agent who may edit primitives.py. The file is past the soft cap. Do not create a sibling. Magic cost is the AP-072 figure already measured. The Fire Rod chest in room 0x58 is unopened. A disclosed write does not pass a Done line that requires the rod to rise in play.",
            "AP-149. Outdoor screen 0x40 did not fall. Indoor room 0x58 did not fall. BigKeyD3 was not entered. Do not repeat those stands. Do not poke $A0.",
            "snes/alttp/primitives.py, the primitive test that covers hookshot, snes/alttp/docs/routes/skull_ROUTE.md, snes/alttp/dungeon/route_data/skull.yaml, snes/alttp/dungeon/bosses/mothula.py, recordings under snes/alttp/recordings/probe_AP-150/"
        ),
        label: "mgr-AP-150",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "swamp",
            "AP-141 only",
            "hookshot() already exists. Use it from SwampPalaceEntry or Whatif_SwampPalaceEntry_parallel. Do not edit primitives.py.",
            "AP-139 north_to_0x38 stays in room 0x28 at (4344,1336) layer 1. Do not retry that push. Do not poke flippers. AP-144 is blocked: hookshot killed at most 7 of 13 puffs and the core stayed at 32. Do not refight Arrghus. A big-key poke does not pass a Done line that requires the key to rise in play.",
            "snes/alttp/docs/routes/swamp_ROUTE.md, snes/alttp/dungeon/route_data/swamp.yaml, recordings under snes/alttp/recordings/probe_AP-141/"
        ),
        label: "mgr-AP-141",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "thieves",
            "AP-159 only",
            "Read the Blind room id from thieves yaml or z3 data. A boss-room load is allowed for the probe, the same way Helmasaur used Scan_pod_boss. Do not invent the room. Do not edit primitives.py. Do not start AP-160 unless the probe shows what hurts Blind.",
            "AP-155, AP-156, AP-157, and AP-158 are blocked. Do not retry the lift, the conveyor scan, the layer-1 chest, or the attic flood. East shutter stayed 0x18. Attic 0x65 and cell 0x45 were not entered.",
            "snes/alttp/docs/routes/thieves_ROUTE.md, snes/alttp/dungeon/route_data/thieves.yaml, snes/alttp/dungeon/bosses/blind.py, recordings under snes/alttp/recordings/probe_AP-159/"
        ),
        label: "mgr-AP-159",
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
log("interior lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, ids that changed from 110 done and 24 blocked, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Finish wave 2 ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
