let meta = #{
    name: "alttp-board",
    description: "Per-dungeon autobot metrics, and a room_engine shrink that refuses a sibling file",
    when_to_use: "After Play_eog9_7. Report every dungeon. Do not boot room 0xB9. Do not split room_engine.py.",
    phases: [
        #{ title: "Board", detail: "twelve route metrics and one room_engine shrink" },
        #{ title: "Report", detail: "one table, boot stays no" },
    ],
};

let metric_schema = #{
    "type": "object",
    "required": ["dungeon", "last_room", "boss", "open_red", "boot"],
    "properties": #{
        "dungeon": #{ "type": "string" },
        "last_room": #{ "type": "string" },
        "boss": #{ "type": "string" },
        "open_red": #{ "type": "string" },
        "boot": #{ "type": "boolean" },
    },
};

let gut_schema = #{
    "type": "object",
    "required": ["lines", "under_cap", "note"],
    "properties": #{
        "lines": #{ "type": "integer" },
        "under_cap": #{ "type": "boolean" },
        "note": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "boot_count"],
    "properties": #{
        "summary": #{ "type": "string" },
        "boot_count": #{ "type": "integer" },
    },
};

fn metric_prompt(slug, route) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are the read-only metric for one dungeon, slug ";
    p += slug;
    p += ".\n";
    p += "Read snes/alttp/docs/routes/";
    p += route;
    p += " and only that dungeon's headings in snes/alttp/docs/PORT_PLAN.md. Use read_file and grep. Do not answer from memory.\n";
    p += "Do not edit any file. Do not boot a ROM. Do not run bd. Do not claim, tick, or block a task. Do not edit PORT_PLAN.md or docs/STATUS.md.\n";
    p += "Return dungeon as the slug. last_room is one room id the route doc says was entered, or none. boss is a quoted score such as 12/12 or 0/12, or none. open_red is one sentence copied from that route doc, or none. boot is false.\n";
    p += "A blocked big key, a claimed fight, and a step 7 approach are not a reason to set boot true.\n";
    return p;
}

let gut_prompt = "";
gut_prompt += "Repo: /home/v/01_projects/11_games/retro. You own snes/alttp/opening_route/room_engine.py and no other file.\n";
gut_prompt += "The file is about 1218 lines. The cap in CODING_STANDARDS.md is about 1000. Every function from in_room through run_room_edge is already called from inside this file. Do not create a new module. Do not move functions into nav.py, scripts/room_engine.py, or any other file to get under the cap. A sibling extract is a failed sitting.\n";
gut_prompt += "Delete only code that has no caller. If you cannot get the file under 1000 lines that way, leave the file byte-for-byte unchanged.\n";
gut_prompt += "Do not boot a ROM. Do not edit maps, PORT_PLAN.md, docs/STATUS.md, primitives.py, or docs/tasks/residual.md. Do not run bd. Do not git commit.\n";
gut_prompt += "If you do edit the file, run from snes/: QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests/test_room_engine_edges.py alttp/tests/test_room_sense.py alttp/tests/test_map_lint.py -q. Those tests must pass. If they fail, restore the file.\n";
gut_prompt += "Use wc -l on the file before you return. Return lines (the wc count), under_cap (true only if the file is under 1000 and the tests passed), and note (one sentence: what you deleted, or that every helper is live and the file was left unchanged).\n";

phase("Board");
let jobs = [
    #{
        prompt: gut_prompt,
        label: "gut-room-engine",
        capability_mode: "all",
        output_schema: gut_schema,
    },
];
let specs = [
    #{ slug: "eastern", route: "eastern_ROUTE.md" },
    #{ slug: "desert", route: "desert_ROUTE.md" },
    #{ slug: "hera", route: "hera_ROUTE.md" },
    #{ slug: "tower", route: "castle_tower_ROUTE.md" },
    #{ slug: "pod", route: "pod_ROUTE.md" },
    #{ slug: "swamp", route: "swamp_ROUTE.md" },
    #{ slug: "skull", route: "skull_ROUTE.md" },
    #{ slug: "thieves", route: "thieves_ROUTE.md" },
    #{ slug: "ice", route: "ice_ROUTE.md" },
    #{ slug: "mire", route: "mire_ROUTE.md" },
    #{ slug: "turtle", route: "turtle_rock_ROUTE.md" },
    #{ slug: "gt", route: "gt_ROUTE.md" },
];
for spec in specs {
    jobs.push(#{
        prompt: metric_prompt(spec.slug, spec.route),
        label: "metric:" + spec.slug,
        capability_mode: "read-only",
        output_schema: metric_schema,
    });
}
let results = parallel(jobs);

let quoted = [];
let gut_note = "room_engine shrink did not return";
let gut_lines = 0;
for r in results {
    if r == () || r.success != true || type_of(r.output) != "map" {
        continue;
    }
    if r.output.dungeon != () && r.output.dungeon != "" {
        let boot_bit = "no";
        if r.output.boot == true {
            boot_bit = "AGENT SAID YES";
        }
        let line = r.output.dungeon;
        line += " room ";
        line += if r.output.last_room != () { r.output.last_room } else { "none" };
        line += " boss ";
        line += if r.output.boss != () { r.output.boss } else { "none" };
        line += " boot ";
        line += boot_bit;
        line += " | ";
        line += if r.output.open_red != () { r.output.open_red } else { "none" };
        quoted.push(line);
    }
    if r.output.lines != () && r.output.note != () {
        gut_lines = r.output.lines;
        gut_note = r.output.note;
    }
}
log("metric rows " + quoted.len().to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Repo: /home/v/01_projects/11_games/retro. From that directory run PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes uv run python -m alttp.scripts.dungeon_board. Paste that table into your summary.\n";
report_prompt += "Then add these route quotes, one per line. If a quote says AGENT SAID YES, write that the board still forbids the boot. Do not boot any ROM. Do not edit PORT_PLAN.md, docs/STATUS.md, maps, or room_engine.py.\n";
report_prompt += "Room engine note: ";
report_prompt += gut_note;
report_prompt += "\n";
report_prompt += "Route quotes:\n";
for line in quoted {
    report_prompt += line;
    report_prompt += "\n";
}
report_prompt += "Return summary (the table plus the quotes) and boot_count 0.\n";
let report = agent(report_prompt, #{
    label: "board-report",
    capability_mode: "all",
    output_schema: report_schema,
});
let summary = "board report missing";
let boot_count = 0;
if report != () && report.success == true && type_of(report.output) == "map" {
    if report.output.summary != () { summary = report.output.summary; }
    if report.output.boot_count != () { boot_count = report.output.boot_count; }
}
let body = summary;
body += "\n\nroom_engine lines ";
body += gut_lines.to_string();
body += "\n";
body += gut_note;
body += "\n";
let path = write_scratch_file("dungeon_board.md", body);
complete(#{
    path: path,
    summary: summary,
    boot_count: boot_count,
    room_engine_lines: gut_lines,
});
