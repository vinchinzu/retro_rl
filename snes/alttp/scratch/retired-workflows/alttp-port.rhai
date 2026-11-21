let meta = #{
    name: "alttp-port",
    description: "Managers and sub-agents for snes/alttp/docs/PORT_PLAN.md, one file-owning lane at a time",
    when_to_use: "Work the ALTTP port plan AP-001 through AP-200 without beads. Tick boxes in PORT_PLAN.md are the tracker.",
    phases: [
        #{ title: "Independent", detail: "docs manager plus file-disjoint sub-agents" },
        #{ title: "Plan check", detail: "200 headings still intact" },
        #{ title: "Core lanes", detail: "RAM, spine, rollout" },
        #{ title: "Measure lanes", detail: "ledger, scan, replay, sprite names" },
        #{ title: "Movement", detail: "tiles, walker, primitives, optional TAS sync" },
        #{ title: "Late scans", detail: "tiles, tape scanner, census" },
        #{ title: "Combat", detail: "guards, strike, census" },
        #{ title: "Sanctuary", detail: "castle chain to rung C1" },
        #{ title: "Overworld", detail: "side items until a dungeon prize is required" },
        #{ title: "Dungeons", detail: "twelve dungeon managers, then phase I" },
        #{ title: "Report", detail: "tick counts" },
    ],
};

let worker_schema = #{
    "type": "object",
    "required": ["task", "status", "evidence"],
    "properties": #{
        "task": #{ "type": "string" },
        "status": #{ "type": "string" },
        "evidence": #{ "type": "string" },
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

let gate_schema = #{
    "type": "object",
    "required": ["plan_headings", "dungeons_unblocked"],
    "properties": #{
        "plan_headings": #{ "type": "integer" },
        "dungeons_unblocked": #{ "type": "boolean" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "open_count", "done_count", "blocked_count"],
    "properties": #{
        "summary": #{ "type": "string" },
        "open_count": #{ "type": "integer" },
        "done_count": #{ "type": "integer" },
        "blocked_count": #{ "type": "integer" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. You are one worker on the ALTTP port.\n";
    p += "Tracker: snes/alttp/docs/PORT_PLAN.md only. Do not run bd. Do not create or close beads. Ignore the alttp-session bead loop for this sitting.\n";
    p += "Read the rules at the top of PORT_PLAN.md once (through the phase map). Then read only the task you are about to do, plus snes/alttp/AGENTS.md and the files that task names. Do not read other tasks.\n";
    p += "Claim, finish, and block only with this script, which holds a file lock:\n";
    p += "  python3 /home/v/.grok/alttp_port_claim.py claim AP-000 YOUR-AGENT\n";
    p += "  python3 /home/v/.grok/alttp_port_claim.py done AP-000 \"short evidence\"\n";
    p += "  python3 /home/v/.grok/alttp_port_claim.py block AP-000 \"reason and evidence path\"\n";
    p += "  python3 /home/v/.grok/alttp_port_claim.py note AP-000 \"one lesson the next agent needs\"\n";
    p += "If claim prints claimed-by-other or already-done, leave that task alone.\n";
    p += "Do not rewrite PORT_PLAN.md in an editor. Do not git add, git commit, or git checkout. Do not revert unrelated dirty files (harvest, sm, beads, refs).\n";
    p += "Do not edit docs/STATUS.md unless the task is AP-082, a dungeon step 8, AP-199, or AP-200, and only with the evidence that task names. A save-state pin is not continuous.\n";
    p += "No RAM writes on a natural or continuous run. Do not poke $F3CC, keys, rupees, HP, or items in a run you will quote. Do not overwrite recordings/verified_tip_run.json on a red run.\n";
    p += "Geometry stays in alttp/maps json. Do not copy approach xy into Python. Soft max about 1000 lines; merge into the owning module, do not split a sibling file.\n";
    p += "z3-json-data and a live scan beat hint lines. Prefer snes/alttp/refs/z3-json-data when that checkout exists, else snes/alttp/z3-json-data. Do not commit refs, do not delete the committed dump, do not download.\n";
    p += "RAM addresses in the plan are candidates until you verify them on a named pin. If a hint is wrong, fix that hint line and add a note.\n";
    p += "Leave the AP-002 Owner ruling line blank. Do not invent a lookahead ruling for ALTTP.\n";
    p += "Commands that boot the ROM use cwd snes/ and SDL_VIDEODRIVER=dummy. Pytest uses QT_QPA_PLATFORM=offscreen. Docs tests run from the repo root: uv run pytest tests/test_docs.py -q. One emulator in your process. Do not kill other Python processes.\n";
    p += "Touch only the files the task lists. Two honest tries, then block. Do not weaken the Done check.\n";
    p += "Use read_file, grep, and the shell. Do not answer from memory.\n";
    return p;
}

fn worker_prompt(task, agent, extra) {
    let p = rules();
    p += "You implement exactly one task: " + task + ". Your agent name is " + agent + ".\n";
    p += extra;
    p += "\nReturn task, status (done, blocked, or skipped), and evidence (the Done command or the block reason).\n";
    return p;
}

fn lane_prompt(agent, tasks, extra) {
    let p = rules();
    p += "You are the manager for agent name " + agent + ".\n";
    p += "Do these tasks in this order, yourself: " + tasks + ".\n";
    p += extra;
    p += "\nBefore each task, re-read its Needs line. Skip a task that is already [x]. If it is [~] or [!] for someone else, stop the lane.\n";
    p += "If a Needs id is not [x], stop the lane. Set waiting_on to that id. Do not mark the task blocked.\n";
    p += "If you block a task, continue a later task only when its Needs do not include the blocked id. If you are unsure, stop.\n";
    p += "Re-read a source file immediately before you edit it. Do not remove functions another task added.\n";
    p += "Return completed (AP ids you marked [x] this sitting), waiting_on (one id or empty), blocked (one id or empty).\n";
    return p;
}

phase("Independent");
let jobs = [];
jobs.push(#{
    prompt: lane_prompt(
        "mgr-docs",
        "AP-001, AP-002, AP-004",
        "AP-001 may remove only the non-goal line Full dungeon ladder or Ganon credits from snes/alttp/docs/plan.md, and add one relative link to PORT_PLAN.md. Keep every other plan.md rule. AP-002 writes snes/alttp/docs/ASSIST_CONTRACT.md from the Zelda I contract: Clean, Survival, what-if pins, and rollout lookahead disclosed as rollout_restores. State that Zelda I lookahead permission is not a ruling for ALTTP, and leave an Owner ruling line blank. Add one pointer line to snes/alttp/AGENTS.md without deleting the bd line. AP-004 writes the dungeon route template only. After AP-001 and AP-002, run the docs tests once from the repo root."
    ),
    label: "mgr-docs",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-003", "ap-003", "Own only the clean-ladder files the task names. next_open must return C1."),
    label: "ap-003",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-005", "ap-005", "Own ram.py, the new inventory test, and the inventory section of ram_map.md. Mark each field verified_on a pin or unverified."),
    label: "ap-005",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-011", "ap-011", "Sprites live in primitives.py, the composer. Do not start a new sprite module."),
    label: "ap-011",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-012", "ap-012", "Own sprite_ids.py only."),
    label: "ap-012",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-019", "ap-019", "Own the tape writer and its unit test. Reuse retro_harness tape code if it fits. Do not boot a long ROM run for the unit test."),
    label: "ap-019",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-040", "ap-040", "Extract dungeon locations from z3-json-data. Do not commit the refs checkout."),
    label: "ap-040",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-044", "ap-044", "Own the map lint the task names. Do not change room behaviour that the task does not mention."),
    label: "ap-044",
    capability_mode: "all",
    output_schema: worker_schema,
});
jobs.push(#{
    prompt: worker_prompt("AP-054", "ap-054", "Docstring and a measured budget only. Do not implement Rollout.fan in this task."),
    label: "ap-054",
    capability_mode: "all",
    output_schema: worker_schema,
});
let wave1 = parallel(jobs);
let wave1_done = 0;
if wave1 != () {
    for slot in wave1 {
        if slot != () && slot.success {
            wave1_done += 1;
        }
    }
}
log("independent slots returned " + wave1_done.to_string());

phase("Plan check");
let accept_prompt = rules();
accept_prompt += "You are the plan-file manager. Do not start new feature work.\n";
accept_prompt += "Run: python3 /home/v/.grok/alttp_port_claim.py check\n";
accept_prompt += "Set plan_headings to the integer it prints. If the command fails or the integer is not 200, set plan_headings to 0 and set dungeons_unblocked to false.\n";
accept_prompt += "Also set dungeons_unblocked to false in this phase. A later manager decides the dungeon gate.\n";
accept_prompt += "If the heading count is 200 and one of AP-001, AP-002, AP-003, AP-004, AP-005, AP-011, AP-012, AP-019, AP-040, AP-044, AP-054 is still open, implement that single task with the same rules, then re-run check. Do not implement any other id.\n";
let accept = agent(accept_prompt, #{
    label: "mgr-plan",
    capability_mode: "all",
    output_schema: gate_schema,
});
let headings = 0;
if accept != () && accept.success && accept.output != () && type_of(accept.output) == "map" && accept.output.plan_headings != () {
    headings = accept.output.plan_headings;
}
if headings != 200 {
    log("plan check did not report 200 headings; later lanes still run and stop on unmet needs");
} else {
    log("plan headings " + headings.to_string());
}

phase("Core lanes");
let core = [];
core.push(#{
    prompt: lane_prompt(
        "mgr-ram",
        "AP-007, AP-009, AP-010, AP-013, AP-008, AP-020",
        "You own ram.py and ram_map.md. Do not edit primitives.py, rollout.py, hops.py, or tape.py. AP-005 is already owned by ap-005; skip it if it is [x], and wait if it is not. AP-020 needs AP-019. Do not play the Tompa movie."
    ),
    label: "mgr-ram",
    capability_mode: "all",
    output_schema: lane_schema,
});
core.push(#{
    prompt: lane_prompt(
        "mgr-spine",
        "AP-022, AP-023, AP-024, AP-025, AP-026, AP-027, AP-028, AP-029",
        "You own tape.py, session.py, hops.py, run_spine.py, audit_pins.py, stage_replay.py, and eval_offsets.py, only as each task lists them. Do not edit ram.py, primitives.py, or rollout.py. Do not take AP-030."
    ),
    label: "mgr-spine",
    capability_mode: "all",
    output_schema: lane_schema,
});
core.push(#{
    prompt: lane_prompt(
        "mgr-roll",
        "AP-055",
        "Own rollout.py after AP-054. Implement Rollout.fan only. Do not take AP-056 or PolicyGuard."
    ),
    label: "mgr-roll",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(core);

phase("Measure lanes");
let measure = [];
measure.push(#{
    prompt: lane_prompt(
        "mgr-ledger",
        "AP-014, AP-015, AP-016, AP-017, AP-018",
        "You own ledger.py, room_timer.py, and the opening-spine ledger flag. Do not edit ram.py."
    ),
    label: "mgr-ledger",
    capability_mode: "all",
    output_schema: lane_schema,
});
measure.push(#{
    prompt: lane_prompt(
        "mgr-scan",
        "AP-031, AP-032, AP-034, AP-035, AP-036, AP-041, AP-042",
        "You own pin_probe.py, the door section of ram_map.md, whatif_pin.py, and make_entry_pin.py. Delete a scratch probe only when the task names that file. Do not edit rollout.py or nav.py."
    ),
    label: "mgr-scan",
    capability_mode: "all",
    output_schema: lane_schema,
});
measure.push(#{
    prompt: lane_prompt(
        "mgr-replay",
        "AP-021",
        "Own replay_tape.py. The Done check is a verified-tip replay with desynced == 0. Do not overwrite verified_tip_run.json."
    ),
    label: "mgr-replay",
    capability_mode: "all",
    output_schema: lane_schema,
});
measure.push(#{
    prompt: lane_prompt(
        "mgr-sprite",
        "AP-061",
        "Own sprite_ids.py. Vulnerability table only."
    ),
    label: "mgr-sprite",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(measure);

phase("Movement");
let move = [];
move.push(#{
    prompt: lane_prompt(
        "mgr-move",
        "AP-045, AP-046, AP-047, AP-048, AP-049, AP-050, AP-051, AP-052, AP-053, AP-056",
        "You own tilemap.py, walk_physics.py, nav.py, and the room_engine walker hook. Do not edit ram_map.md or rollout.py until AP-056, and then only add Rollout.walk. Do not edit pin_probe.py."
    ),
    label: "mgr-move",
    capability_mode: "all",
    output_schema: lane_schema,
});
move.push(#{
    prompt: lane_prompt(
        "mgr-prim",
        "AP-030, AP-065, AP-066, AP-067",
        "You own new primitives in primitives.py: save_and_quit, Y-item select, bottle guard, bomb, lift, throw, push. Do not edit nav.py, rollout.py, or ram_map.md."
    ),
    label: "mgr-prim",
    capability_mode: "all",
    output_schema: lane_schema,
});
parallel(move);

phase("Late scans");
let late = agent(lane_prompt(
    "mgr-scan2",
    "AP-033, AP-037, AP-038, AP-039, AP-043",
    "You own the remaining pin_probe flags, scan_tape.py, room_census.py, and the ledger census book. Do not edit nav.py."
), #{ label: "mgr-scan2", capability_mode: "all", output_schema: lane_schema });
if late == () || late.success != true {
    log("scan2 manager returned no success");
}

phase("Combat");
let combat = agent(lane_prompt(
    "mgr-combat",
    "AP-057, AP-058, AP-059, AP-060, AP-062, AP-063, AP-064, AP-068, AP-069, AP-070, AP-071, AP-072",
    "You own PolicyGuard and strike in rollout.py, clear policy in room_engine, assist.py, and the census reports. Do not edit nav.py or the inventory reader. Survival assist stays dev-only and disclosed. Do not take AP-061 or AP-065."
), #{ label: "mgr-combat", capability_mode: "all", output_schema: lane_schema });
if combat == () || combat.success != true {
    log("combat manager returned no success");
}

phase("Sanctuary");
let sanctuary = agent(lane_prompt(
    "mgr-sanctuary",
    "AP-073, AP-074, AP-075, AP-076, AP-077, AP-078, AP-079, AP-080, AP-081, AP-082",
    "Push the verified tip from room 0x50 through the big key, Zelda's cell, the escort, and Sanctuary. The known red is the 0x81 gold cell door. Read the castle notes already in PORT_PLAN and snes/alttp/docs/tasks/residual.md before AP-073. Do not call a pin green continuous. AP-082 is the only task here that may edit STATUS.md, and only if the power-on run meets the task's evidence."
), #{ label: "mgr-sanctuary", capability_mode: "all", output_schema: lane_schema });
if sanctuary == () || sanctuary.success != true {
    log("sanctuary manager returned no success");
}

phase("Overworld");
let ow_prompt = lane_prompt(
    "mgr-overworld",
    "AP-083, AP-084, AP-085, AP-086, AP-087, AP-088, AP-089, AP-090, AP-091, AP-092, AP-093, AP-094, AP-095, AP-096, AP-097, AP-098",
    "Stop at the first task whose Needs are not all [x]. Dungeon prizes and later items are allowed to wait. Do not mark a waiting task blocked. Do not edit dungeon boss modules."
);
let ow = agent(ow_prompt, #{ label: "mgr-overworld", capability_mode: "all", output_schema: lane_schema });
let ow_latest = "";
if ow != () && ow.success && ow.agent_id != () {
    ow_latest = ow.agent_id;
}

phase("Dungeons");
let gate_prompt = "";
gate_prompt += "Read snes/alttp/docs/PORT_PLAN.md tick boxes with the shell, not from memory.\n";
gate_prompt += "Run python3 /home/v/.grok/alttp_port_claim.py check and set plan_headings to that integer, or 0 if it fails.\n";
gate_prompt += "Set dungeons_unblocked true only if AP-027, AP-035, AP-036, AP-042, and AP-044 are all marked [x]. Otherwise false.\n";
gate_prompt += "Do not edit files.\n";
let gate = agent(gate_prompt, #{
    label: "mgr-dungeon-gate",
    capability_mode: "execute",
    output_schema: gate_schema,
});
let dungeons_ready = false;
if gate != () && gate.success && type_of(gate.output) == "map" && gate.output.dungeons_unblocked == true {
    dungeons_ready = true;
}
if dungeons_ready {
    let dungeons = [
        #{ lo: "AP-099", hi: "AP-106", name: "Eastern Palace" },
        #{ lo: "AP-107", hi: "AP-114", name: "Desert Palace" },
        #{ lo: "AP-115", hi: "AP-122", name: "Tower of Hera" },
        #{ lo: "AP-123", hi: "AP-130", name: "Agahnim's Tower" },
        #{ lo: "AP-131", hi: "AP-138", name: "Palace of Darkness" },
        #{ lo: "AP-139", hi: "AP-146", name: "Swamp Palace" },
        #{ lo: "AP-147", hi: "AP-154", name: "Skull Woods" },
        #{ lo: "AP-155", hi: "AP-162", name: "Thieves' Town" },
        #{ lo: "AP-163", hi: "AP-170", name: "Ice Palace" },
        #{ lo: "AP-171", hi: "AP-178", name: "Misery Mire" },
        #{ lo: "AP-179", hi: "AP-186", name: "Turtle Rock" },
        #{ lo: "AP-187", hi: "AP-194", name: "Ganon's Tower" },
    ];
    for spec in dungeons {
        let dp = lane_prompt(
            "mgr-" + spec.lo,
            spec.lo + " through " + spec.hi + " in order",
            "You are the dungeon manager for " + spec.name + " only. Read the dungeon task template (the eight steps) and your dungeon section. Do not read other dungeons. Steps 1-6 can finish while a later Need is open; stop at the first unmet Need and set waiting_on. Never call a pin natural. Boss step 6 needs at least 10 of 12 RNG offsets. Hint lines are not authority. You may edit hops.py only by re-reading it and adding your own rows. You may fix a wrong hint inside your own dungeon section. Do not edit another dungeon's boss module or route doc."
        );
        let dres = agent(dp, #{
            label: "mgr-" + spec.lo,
            capability_mode: "all",
            output_schema: lane_schema,
        });
        if dres == () || dres.success != true {
            log("dungeon manager failed " + spec.name);
        }
        if ow_latest != "" {
            let cont = agent(
                "Continue the overworld lane. Re-read PORT_PLAN ticks. Do the next open tasks from AP-083 through AP-098 whose Needs are all [x]. Stop at the first unmet Need without blocking it. Do not redo [x] tasks.",
                #{ label: "mgr-overworld-next", capability_mode: "all", output_schema: lane_schema, resume_from: ow_latest }
            );
            if cont != () && cont.success && cont.agent_id != () {
                ow_latest = cont.agent_id;
            }
        }
    }
    let finale = agent(lane_prompt(
        "mgr-finale",
        "AP-198, AP-195, AP-196, AP-197, AP-199, AP-200",
        "If a Needs id is not [x], stop with waiting_on. Do not block a task just because an earlier dungeon is unfinished. AP-198: if stage reports do not exist yet, wait instead of blocking. AP-199 and AP-200 may edit STATUS.md only when their evidence is real. AP-200 may edit docs/manifests/alttp.yaml and regenerate the game matrix."
    ), #{ label: "mgr-finale", capability_mode: "all", output_schema: lane_schema });
    if finale == () || finale.success != true {
        log("finale manager returned no success");
    }
} else {
    log("dungeon gate closed; step-1 tools are not all done");
}

phase("Report");
let tail = [];
let report_prompt = "";
report_prompt += "Count PORT_PLAN ticks by running Python against /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md.\n";
report_prompt += "Count heading lines only. open_count is lines whose box is a space. done_count is [x]. blocked_count is [!].\n";
report_prompt += "summary: one sentence with those three numbers, then the blocked AP ids and the lowest open AP id. Do not edit files.\n";
tail.push(#{
    prompt: report_prompt,
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
tail.push(#{
    prompt: lane_prompt(
        "mgr-tas",
        "AP-006",
        "Optional oracle, off the critical path. Own tas_sync.py. You may append one section to ram_map.md; re-read it first and do not rewrite other sections. Two honest tries. If the movie header or start state will not load, block AP-006. Do not play all 274k frames. No other task needs this one."
    ),
    label: "mgr-tas",
    capability_mode: "all",
    output_schema: lane_schema,
});
let tail_results = parallel(tail);
let report = ();
if tail_results != () {
    report = tail_results[0];
}
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Lanes finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
