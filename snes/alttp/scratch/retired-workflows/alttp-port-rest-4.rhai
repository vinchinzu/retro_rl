let meta = #{
    name: "alttp-port-rest-4",
    description: "Vitreous and Trinexx policies, plus the laser check and the red-mail room",
    when_to_use: "After alttp-port-rest-3. Both boss probes are done. Do not retry the blocked walks.",
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
    p += "\nRe-read the heading. Skip it if it is [x] or [!]. A step is done only when its Done line passes. Two honest tries, then block. Do not weaken the Done line. A boss policy needs at least 10 of 12 RNG offsets. A poke does not pass a line that requires the item to rise in play. Hint lines are not authority.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Do not edit primitives.py, make_entry_pin.py, or bosses/agahnim.py. Re-read a file immediately before you edit it. Soft max about 1000 lines. Do not extract a sibling module.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
    return p;
}

phase("Lanes");
let jobs = [
    #{
        prompt: lane_prompt(
            "mire",
            "AP-176 only",
            "Start from Scan_mire_boss, room 0x90, kind poke. Body sprite 0xBD HP 128. Thirteen eyes 0xBE HP 48. Measured: a spin 16 to 24px south flashes and drops one eye from 48 to 44. A fighter sword leaves an eye at 48. Body offsets 0 through 32 do not drop the body. Clearing the eyes with a poke does not count. Do not invent a second damage rule.",
            "The Mire lobby pit, the somaria block that left $0646 at 0, and any walk that starts in room 0x98.",
            "snes/alttp/dungeon/bosses/vitreous.py, snes/alttp/docs/routes/mire_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-176/"
        ),
        label: "mgr-AP-176",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "turtle",
            "AP-184 only",
            "Start from Scan_turtle_rock_boss, room 0xA4, kind poke. Heads 0xCB, 0xCC, and 0xCD start at HP 40 and AI 0. Idle 1800 did not make them born. Measured on an unsaved camera: a poked fire rod drops the ice head from 40 to 36 when its AI is 3. An ice rod flashes the fire head at AI 3 and leaves it at 40. The policy has to reach AI 3. Do not poke the AI byte. The fire-rod byte on that probe was a disclosed write. Say so in the evidence.",
            "The Turtle Rock rail. ride_platform spawned sprite 0xED and $02F5 stayed 0. UP stuck at (3448,7016). Do not retry that ride.",
            "snes/alttp/dungeon/bosses/trinexx.py, snes/alttp/docs/routes/turtle_rock_ROUTE.md, recordings under snes/alttp/recordings/probe_AP-184/"
        ),
        label: "mgr-AP-184",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "lasers",
            "AP-182 only",
            "Read which room holds the laser eyes from turtle rock yaml or z3. Load that room the way the Trinexx probe loaded room 0xA4. Measure which facings stop the beam. Do not edit turtle_rock_ROUTE.md. The Trinexx lane owns it. If you cannot quote a room id, leave AP-182 open and do not boot.",
            "AP-179 and AP-181. Do not stand on the 0xD6 rail and do not call ride_platform.",
            "recordings under snes/alttp/recordings/probe_AP-182/"
        ),
        label: "mgr-AP-182",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "gt",
            "AP-190 only",
            "Read the red-mail room id from gt yaml or z3. Load that room. Do not walk there from room 0x6B. If no file quotes the room id, leave AP-190 open and do not boot.",
            "AP-189 and AP-191. Room 0x1C was not entered. Agahnim 2 showed one body and no clone. Do not retry either.",
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
log("rest-4 lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, the mark on AP-176, AP-182, AP-184, and AP-190, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Rest wave 4 ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
