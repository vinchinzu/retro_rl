let meta = #{
    name: "alttp-eog9-lobby",
    description: "Take the existing Eastern north hop from Play_eog9_6 into room 0xB9",
    when_to_use: "After alttp-eog9-keys. route_goal_ok is true. The lobby popos are not claimed.",
    phases: [
        #{ title: "Lobby", detail: "Play_eog9_6 through the north door" },
        #{ title: "Report", detail: "room 0xB9 glance" },
    ],
};

let lane_schema = #{
    "type": "object",
    "required": ["closed", "glance", "blocker"],
    "properties": #{
        "closed": #{ "type": "boolean" },
        "glance": #{ "type": "string" },
        "blocker": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "lobby_closed"],
    "properties": #{
        "summary": #{ "type": "string" },
        "lobby_closed": #{ "type": "boolean" },
    },
};

phase("Lobby");
let lobby_prompt = "";
lobby_prompt += "Repo: /home/v/01_projects/11_games/retro. cwd snes/. Set SDL_VIDEODRIVER=dummy and SDL_AUDIODRIVER=dummy. One emulator. Do not kill other Python processes.\n";
lobby_prompt += "Read maps/room_c9.json and the Eastern route note for the center pot before you boot. Load only custom_integrations/Zelda3-Snes/Play_eog9_6.state. If the live glance is not room 0xC9, indoors, module 0x07, submodule 0x00, control ready, near (4856, 6616), HP 28/32, sword 1, lamp 1, follower 0, stop.\n";
lobby_prompt += "Run the existing room_engine edge room_c9 north_to_0xb9 from that state. The edge already lifts the center pot, fights the three 0x4F popos south of y=6248, and holds UP from about y=6248. A push already at y about 6204 wedges. x at or above 4876 stays in 0xC9. Do not load EasternPalaceEntry. Do not write HP, rupees, keys, sword, lamp, or $F3CC.\n";
lobby_prompt += "room_engine.py is 1218 lines and segment.py is 1000. Do not edit either file. Do not add a module. Two runs of the existing edge, then stop. A death does not close.\n";
lobby_prompt += "Close only when the live step enters room 0xB9, indoors, module 0x07, submodule 0x00, control ready. Record xy, HP, sword, lamp, follower, and keys. Save Play_eog9_7 only after that glance. Replay those buttons from Play_eog9_6 into recordings/eog9_lobby_20261023/. The replay needs zero desync, zero state loads, and zero RAM writes.\n";
lobby_prompt += "Do not tick AP-105. Do not change a checkbox. Do not edit docs/STATUS.md. Do not run bd. Do not git commit. Do not overwrite recordings/eog9_spine_20261023/ or recordings/eog9_door_20261023/.\n";
lobby_prompt += "Overwrite snes/alttp/docs/tasks/residual.md with the glance or the miss, the state file, and the still-open north hall. Keep the Clean Sanctuary sentence for recordings/clean_escape_20261002/.\n";
lobby_prompt += "Return closed, glance, and blocker.\n";
let lobby = agent(lobby_prompt, #{
    label: "mgr-lobby",
    capability_mode: "all",
    output_schema: lane_schema,
});

phase("Report");
let report_prompt = "";
report_prompt += "Repo: /home/v/01_projects/11_games/retro. Read-only. Do not edit files.\n";
report_prompt += "Read snes/alttp/docs/tasks/residual.md and, if present, recordings/eog9_lobby_20261023/replay.json. lobby_closed is true only when that replay boots Play_eog9_6, ends in room 0xB9 with control ready, and reports zero desync and zero RAM writes. A load of EasternPalaceEntry does not count. summary: the glance or the miss. Do not claim AP-105 or credits.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "read-only",
    output_schema: report_schema,
});

let summary = "Lobby hop produced no report.";
let closed = false;
if report != () && report.success == true && type_of(report.output) == "map" {
    if report.output.lobby_closed == true {
        closed = true;
    }
    if report.output.summary != () && type_of(report.output.summary) == "string" {
        let extra = report.output.summary;
        extra.trim();
        if extra != "" {
            summary = extra;
        }
    }
}
complete(#{
    summary: summary,
    lobby_closed: closed,
});
