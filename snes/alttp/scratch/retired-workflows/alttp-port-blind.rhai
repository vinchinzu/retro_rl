let meta = #{
    name: "alttp-port-blind",
    description: "Write the Blind policy from the measured sword phases",
    when_to_use: "After alttp-port-finish-2. AP-159 is done. AP-160 is the open fight.",
    phases: [
        #{ title: "Blind", detail: "10 of 12 from Scan_thieves_boss" },
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

phase("Blind");
let prompt = "";
prompt += "Repo: /home/v/01_projects/11_games/retro. You are the Blind manager. Your only task is AP-160.\n";
prompt += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
prompt += "Re-read the AP-160 heading and its Done line. Skip it if it is already [x] or [!]. A step is done only when that Done line passes. The policy needs at least 10 of 12 RNG offsets. Two honest tries, then block. Do not weaken the check.\n";
prompt += "Start from Scan_thieves_boss, room 0xAC, kind poke. Sprite 0xCE. $0E50 starts at 90. Measured on the probe: a sword 32px south facing UP sets $0EA0=14 at frame 10, and at frame 14 $0E50 goes 90 to 128 while $0F90 goes 0 to 1. Three swords walk $0F90 through 1, then 2, then 0, and head slot 15 appears with $0D90=2. Hammer 24px north leaves $0F90 at 1. A bomb for 200 frames leaves $0F90 at 0. Idle 1800 cycles $0DB0 through 1, 2, 3, 4 and HP stays 90. Type 0x41 appears at frames 75, 83, and 547. Read nes/zelda_i/dungeon/gleeok.py for detached-head tracking. Build snes/alttp/dungeon/bosses/blind.py from those measurements. Do not invent a new hit rule.\n";
prompt += "Do not retry AP-155, AP-156, AP-157, or AP-158. Do not walk the attic, the maiden cell, or the east shutter. Do not edit primitives.py. You may edit snes/alttp/dungeon/bosses/blind.py, snes/alttp/docs/routes/thieves_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-160/. Re-read a file immediately before you edit it. Soft max about 1000 lines. Do not extract a sibling module.\n";
prompt += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Use read_file, grep, and the shell. Do not answer from memory.\n";
prompt += "Return completed, waiting_on (one id or empty), and blocked (one id or empty).\n";
let fight = agent(prompt, #{
    label: "mgr-AP-160",
    capability_mode: "all",
    output_schema: lane_schema,
});
if fight != () && fight.success {
    log("blind lane returned");
} else {
    log("blind lane returned no success");
}

phase("Report");
let report_prompt = "";
report_prompt += "Count AP headings in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. If that tape is absent, end_to_end is false. summary: the three counts, end_to_end, whether AP-160 is x or !, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Blind wave ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
