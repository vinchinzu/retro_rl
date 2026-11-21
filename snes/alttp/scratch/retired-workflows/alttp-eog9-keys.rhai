let meta = #{
    name: "alttp-eog9-keys",
    description: "Re-score the Eastern power-on tape if the only inventory miss is the outdoor key sentinel",
    when_to_use: "After alttp-eog9-door. The lobby glance is real. route_goal_ok is false.",
    phases: [
        #{ title: "Score", detail: "sentinel check, then one replay" },
        #{ title: "Report", detail: "goal bit and glance" },
    ],
};

let score_schema = #{
    "type": "object",
    "required": ["changed", "route_goal_ok", "differing", "glance"],
    "properties": #{
        "changed": #{ "type": "boolean" },
        "route_goal_ok": #{ "type": "boolean" },
        "differing": #{ "type": "string" },
        "glance": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "route_goal_ok"],
    "properties": #{
        "summary": #{ "type": "string" },
        "route_goal_ok": #{ "type": "boolean" },
    },
};

phase("Score");
let score_prompt = "";
score_prompt += "Repo: /home/v/01_projects/11_games/retro. cwd snes/. Set SDL_VIDEODRIVER=dummy and SDL_AUDIODRIVER=dummy. Pytest uses QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
score_prompt += "Read snes/alttp/opening_route/segment.py approach_eastern_palace and recordings/eog9_spine_20261023/eog9_spine.json segment eastern_approach. The tape recordings/eog9_spine_20261023/eog9_spine.tape.npz replayed 46646 frames to room 0xC9 (4856, 6616), module 0x07, control ready, HP 28/32, sword 1, lamp 1, follower 0, desync 0, state loads 0, RAM writes 0, survival write_count 0. eastern_lobby is true. inventory_preserved is false. route_goal_ok is false. The room log on the last step changes only hp 32 to 28 and keys 255 to 0. That log does not list every inventory field.\n";
score_prompt += "Do not run a new controller search. Do not load EasternPalaceEntry. Do not overwrite recordings/eog9_hedge_20261001/, recordings/eog9_door_20261023/, recordings/clean_escape_20261002/, or verified_tip_run.json. Do not tick AP-105. Do not change a checkbox. Do not run bd. Do not git commit.\n";
score_prompt += "First print every inventory field that differs across that Eastern segment, excluding hp. If any name other than keys differs, or if keys is not exactly 255 on one side and 0 on the other, stop. Leave the predicate and the tests alone. changed is false.\n";
score_prompt += "If keys 255 versus 0 is the only difference, treat that pair as the outdoor blank sentinel. Any other key change still fails inventory_preserved. segment.py is 1000 lines. Change the existing compare in place. Do not add a module and do not edit overworld.py.\n";
score_prompt += "The test test_eastern_entry_power_on_replay_lands_lobby_with_heart currently asserts inventory_preserved false, segment ok false, and route_goal_ok false. Update those three assertions to the new rule. Keep the lobby, heart, desync, load, and write assertions.\n";
score_prompt += "Re-replay the existing tape with audit_power_on.py replay. Write the new report beside it as recordings/eog9_spine_20261023/eog9_spine_replay_keys.json. Do not replace eog9_spine_replay.json until route_goal_ok is true and the final glance is still room 0xC9 at (4856, 6616) with HP 28/32. If the new replay matches, replace the old replay json and set the controller segment ok from that result.\n";
score_prompt += "From cwd snes/, run QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests/test_opening_spine.py::test_eastern_entry_power_on_replay_lands_lobby_with_heart alttp/tests/test_opening_spine.py::test_eastern_entry_dispatch_stops_after_the_door_hop -q.\n";
score_prompt += "Overwrite snes/alttp/docs/tasks/residual.md. The door proof is recordings/eog9_door_20261023/replay.json, 162 frames from Play_eog9_5, not a no-step load. State the new route_goal_ok. Keep the Clean Sanctuary sentence for recordings/clean_escape_20261002/.\n";
score_prompt += "Return changed, route_goal_ok, differing (the field list you printed), and glance.\n";
let score = agent(score_prompt, #{
    label: "mgr-keys",
    capability_mode: "all",
    output_schema: score_schema,
});

phase("Report");
let report_prompt = "";
report_prompt += "Repo: /home/v/01_projects/11_games/retro. Read-only. Do not edit files.\n";
report_prompt += "Read recordings/eog9_spine_20261023/eog9_spine_replay.json and, if present, eog9_spine_replay_keys.json. route_goal_ok is true only when a replay json says so and its final room is 0xC9, indoors, control ready, xy within 4 of (4856, 6616), HP 28. summary: that bit, the desync and write counts, and the differing inventory fields named in residual.md. Do not claim credits.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "read-only",
    output_schema: report_schema,
});

let summary = "Key sentinel re-score produced no report.";
let goal = false;
if report != () && report.success == true && type_of(report.output) == "map" {
    if report.output.route_goal_ok == true {
        goal = true;
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
    route_goal_ok: goal,
});
