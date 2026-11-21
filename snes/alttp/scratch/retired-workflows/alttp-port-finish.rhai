let meta = #{
    name: "alttp-port-finish",
    description: "Next ALttP completion wave: Skull entrance, hammer, hookshot, then the fights those unlock",
    when_to_use: "After alttp-port-dungeons. The bot does not play end to end. Do not repeat closed probes.",
    phases: [
        #{ title: "Tools", detail: "Skull pin and the two missing primitives" },
        #{ title: "Dungeons", detail: "steps those tools unlock" },
        #{ title: "Report", detail: "ticks and whether a credits tape exists" },
    ],
};

let pin_schema = #{
    "type": "object",
    "required": ["created", "missing", "note"],
    "properties": #{
        "created": #{ "type": "array", "items": #{ "type": "string" } },
        "missing": #{ "type": "array", "items": #{ "type": "string" } },
        "note": #{ "type": "string" },
    },
};

let tool_schema = #{
    "type": "object",
    "required": ["hammer", "hookshot", "note"],
    "properties": #{
        "hammer": #{ "type": "string" },
        "hookshot": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
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

phase("Tools");
let pin_prompt = "";
pin_prompt += "Repo: /home/v/01_projects/11_games/retro. You are mgr-pins. You create missing dungeon-start pins. You do not implement dungeon steps AP-134 through AP-194.\n";
pin_prompt += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md.\n";
pin_prompt += "Never write $F3CC. Provenance must say poke. A pin is not natural and not continuous. Do not overwrite recordings/verified_tip_run.json.\n";
pin_prompt += "Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, PalaceOfDarknessEntry, SwampPalaceEntry, ThievesTownEntry, HyruleCastleGrounds, Scan_*, Play_*, or any existing Whatif_* pin.\n";
pin_prompt += "Read snes/alttp/scripts/make_entry_pin.py before you edit it. rebuild() calls _today_room and refuses a pin that does not already exist. A direct $A0=0x58 poke was rejected: the overlay stayed Hyrule Castle. Do not repeat that poke.\n";
pin_prompt += "Skull Woods is the one slug with a quoted entrance byte. snes/alttp/docs/tasks/SkullWoodsEntry-residual.md says Archipelago entrance 0x29 is the southeast skull, indoor room 0x58, overworld screen 0x40. Use trigger_entrance with 0x29, which writes $010E and the pre-dungeon module. You may add a first-create path to make_entry_pin.py. Save SkullWoodsEntry only if pin_probe shows indoors, module 0x07, room 0x58, and a dungeon id that is not the castle. If the overlay is still the castle grounds, do not keep the pin and do not mark AP-147 done.\n";
pin_prompt += "For castle_tower, mire, turtle_rock, and gt: add an EntrySpec only when a file in this repo quotes the $010E byte the same way the Skull residual quotes 0x29. z3 node ids are not entrance bytes. The last search found no nodeType entrance. Do not repeat that search and do not invent a byte. If no file quotes one, leave the slug missing and do not boot.\n";
pin_prompt += "cwd for ROM commands is snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Re-read a file immediately before you edit it.\n";
pin_prompt += "Return created (pin names pin_probe showed indoors), missing (slugs still without a pin), and note (entrance byte, room, and module for each pin you saved).\n";
let tool_prompt = "";
tool_prompt += "Repo: /home/v/01_projects/11_games/retro. You are mgr-tools. You own snes/alttp/primitives.py for this wave. Nobody else may edit it.\n";
tool_prompt += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
tool_prompt += "Do AP-134, then AP-142. Re-read each heading first. Skip a heading that is already [x]. A step is done only when its own Done line passes. Two honest tries, then block that step. Do not weaken the Done line.\n";
tool_prompt += "AP-134 is the hammer primitive: pegs and stun. AP-142 is the hookshot primitive: reach, pull, and a target across a gap. select_y_item already lists both names. place_bomb and shoot_arrow are the pattern. Add the functions in primitives.py. The file is already past the soft line cap. Do not create a sibling module. Keep each function small.\n";
tool_prompt += "You may also edit the primitive test file that already covers place_bomb or shoot_arrow, snes/alttp/docs/routes/pod_ROUTE.md, and snes/alttp/docs/routes/swamp_ROUTE.md. Yaml edits are hint fixes only. Do not edit boss files, make_entry_pin.py, or another dungeon.\n";
tool_prompt += "Do not retry these closed facts: PoD big key AP-133 stayed 512 behind wall 0x01. Swamp north_to_0x38 stays in room 0x28. Sword at Arrghus landing (3320,472) did not drop core HP 32. A poke does not pass a Done line that requires the item to rise in play.\n";
tool_prompt += "Prove hammer on a live peg and hookshot on a live target. Name the pin, the sprite or tile, and the RAM change. cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen.\n";
tool_prompt += "Use read_file, grep, and the shell. Do not answer from memory.\n";
tool_prompt += "Return hammer (done, blocked, or open), hookshot (done, blocked, or open), and note (one paragraph of the live evidence).\n";
let tools = parallel([
    #{
        prompt: pin_prompt,
        label: "mgr-pins",
        capability_mode: "all",
        output_schema: pin_schema,
    },
    #{
        prompt: tool_prompt,
        label: "mgr-tools",
        capability_mode: "all",
        output_schema: tool_schema,
    },
]);
let pin_note = "{}";
let tool_note = "{}";
if tools != () {
    if tools[0] != () && tools[0].success && type_of(tools[0].output) == "map" {
        pin_note = json_encode(tools[0].output);
        log("pin manager returned a report");
    } else {
        log("pin manager returned no success");
    }
    if tools[1] != () && tools[1].success && type_of(tools[1].output) == "map" {
        tool_note = json_encode(tools[1].output);
        log("tool manager returned a report");
    } else {
        log("tool manager returned no success");
    }
}

phase("Dungeons");
fn lane_prompt(name, tasks, pin_rule, closed, files, pins_json, tools_json) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are dungeon manager ";
    p += name;
    p += ". Tasks: ";
    p += tasks;
    p += ".\n";
    p += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git commit. Do not edit docs/STATUS.md. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += "Pin report JSON: ";
    p += pins_json;
    p += "\nTool report JSON: ";
    p += tools_json;
    p += "\n";
    p += pin_rule;
    p += "\n";
    p += "Closed, do not repeat: ";
    p += closed;
    p += "\n";
    p += "Re-read each heading. Skip [x]. Skip [!] and [~] owned by someone else. A step is done only when its Done line passes. Two honest tries, then block that step. Do not weaken a Done line. A poke does not pass a line that requires the item to rise in play. Boss step 6 needs at least 10 of 12 RNG offsets. Hint lines are not authority. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not download refs.\n";
    p += "You may edit only: ";
    p += files;
    p += ". Do not edit primitives.py, make_entry_pin.py, whatif_pin.py, ram.py, nav.py, or another dungeon. Re-read a file immediately before you edit it. Geometry stays in maps json. Do not copy approach xy into Python. Soft max about 1000 lines. If the boss file would pass that cap, stop and set waiting_on instead of extracting a sibling.\n";
    p += "cwd snes/. SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen.\n";
    p += "Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed (AP ids you marked [x]), waiting_on (one id or empty), blocked (one id you marked [!] or empty).\n";
    return p;
}
let jobs = [
    #{
        prompt: lane_prompt(
            "skull",
            "AP-147 then later open Skull steps through AP-152",
            "Boot only if the pin report created SkullWoodsEntry and pin_probe shows room 0x58 module 0x07. If it did not, set waiting_on AP-147 and do not boot. Do not poke $A0.",
            "The $A0=0x58 poke kept the castle overlay. Entrance 0x2A is the boss hut, not this pin.",
            "snes/alttp/docs/routes/skull_ROUTE.md, snes/alttp/dungeon/route_data/skull.yaml, snes/alttp/dungeon/bosses/mothula.py, maps for rooms your tasks name, recordings under snes/alttp/recordings/probe_AP-147/",
            pin_note,
            tool_note
        ),
        label: "mgr-AP-147",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "pod",
            "AP-135 then AP-136 if the probe shows what hurts Helmasaur",
            "Start from Whatif_PalaceOfDarknessEntry_parallel or PalaceOfDarknessEntry. A boss-room load is allowed for the probe only, the same way Arrghus used Scan_swamp_boss. Read the Helmasaur room id from the yaml or z3 data. Do not invent it. Do not call that load an entry pin.",
            "AP-133 is blocked. BigKeyD1 stayed shut and $F366 stayed 512. Do not retry that chest. Do not walk the pyramid approach AP-137.",
            "snes/alttp/docs/routes/pod_ROUTE.md, snes/alttp/dungeon/route_data/pod.yaml, snes/alttp/dungeon/bosses/helmasaur.py, maps for the boss room, recordings under snes/alttp/recordings/probe_AP-135/",
            pin_note,
            tool_note
        ),
        label: "mgr-AP-135",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "swamp",
            "AP-144 only",
            "Boot only if the tool report says hookshot is done. The fight pin is Scan_swamp_boss, room 0x06. Use the hookshot primitive. If hookshot is not done, set waiting_on AP-142 and do not boot.",
            "Sword at (3320,472) did not drop core HP 32. Do not repeat that stand. Do not retry north_to_0x38. Do not poke flippers. Leave AP-141 open.",
            "snes/alttp/docs/routes/swamp_ROUTE.md, snes/alttp/dungeon/route_data/swamp.yaml, snes/alttp/dungeon/bosses/arrghus.py, recordings under snes/alttp/recordings/probe_AP-144/",
            pin_note,
            tool_note
        ),
        label: "mgr-AP-144",
        capability_mode: "all",
        output_schema: lane_schema,
    },
    #{
        prompt: lane_prompt(
            "thieves",
            "AP-158 then AP-159 if the attic or the maiden cell is actually entered",
            "Start from ThievesTownEntry or Whatif_ThievesTownEntry_parallel, room 0xDB. Do not claim AP-161.",
            "AP-155, AP-156, and AP-157 are blocked. Do not retry the lift RIGHT at (6088,6903), the conveyor scan, or the layer-1 big-key chest. East shutter stayed 0x18.",
            "snes/alttp/docs/routes/thieves_ROUTE.md, snes/alttp/dungeon/route_data/thieves.yaml, snes/alttp/dungeon/bosses/blind.py, maps for rooms your tasks name, recordings under snes/alttp/recordings/probe_AP-158/",
            pin_note,
            tool_note
        ),
        label: "mgr-AP-158",
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
log("dungeon lanes returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Read /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md and count AP headings with Python. open_count is a space box. done_count is x. blocked_count is !. end_to_end is true only if open_count is 0, blocked_count is 0, and a power-on tape under snes/alttp/recordings/ reaches Ganon credits with zero loads and zero RAM writes. Search recordings for a credits or ganon replay json. If you do not find that tape, end_to_end is false. summary: the three counts, end_to_end, the ids marked done or blocked since the previous 105 done and 20 blocked, and the lowest open id. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Finish wave ended without a report.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
    end_to_end: false,
});
