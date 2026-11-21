let meta = #{
    name: "harvest-pr999",
    description: "Managers for the Harvest perfect-rancher plan, one lane at a time",
    when_to_use: "File PR-001 through PR-100 and run Wave 0, then Wave 1 if the F10 verdict says 999 is reachable without a glitch.",
    phases: [
        #{ title: "File", detail: "one beads manager files PR-001 through PR-100" },
        #{ title: "Wave 0", detail: "eight file-disjoint managers" },
        #{ title: "ROM gates", detail: "F0 through F4 and F6" },
        #{ title: "Economy", detail: "F5 after the herd, cover, and family notes" },
        #{ title: "Verdict", detail: "F10" },
        #{ title: "Beads", detail: "orchestrator closes only finished beads" },
        #{ title: "Wave 1", detail: "lane managers, only if 999 is reachable" },
    ],
};

let file_schema = #{
    "type": "object",
    "required": ["filed", "evidence"],
    "properties": #{
        "filed": #{ "type": "integer" },
        "evidence": #{ "type": "string" },
        "root_id": #{ "type": "string" },
    },
};

let lane_schema = #{
    "type": "object",
    "required": ["completed", "evidence"],
    "properties": #{
        "completed": #{ "type": "array", "items": #{ "type": "string" } },
        "evidence": #{ "type": "string" },
        "blocked": #{ "type": "string" },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["completed", "evidence", "reachable"],
    "properties": #{
        "completed": #{ "type": "array", "items": #{ "type": "string" } },
        "evidence": #{ "type": "string" },
        "reachable": #{ "type": "boolean" },
        "natural_max": #{ "type": "string" },
        "binding_term": #{ "type": "string" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. You are one manager on the Harvest perfect-rancher campaign.\n";
    p += "Read snes/harvest/AGENTS.md, snes/harvest/docs/perfect_rancher/MASTER_PLAN.md, and only the bead sections named below in BEAD_PLAN.md. Use read_file and grep. Do not answer from memory.\n";
    p += "The goal is ranch_mastery 999 at ending 0x25 from one Clean continuous power-on run. Every score term has to be maxed, and two terms are windows: each girl 496-511 hearts, each of the four crops 496-511 shipped.\n";
    p += "Clean route code only. No LiveRamEditor and no --ram-set on a route. RAM edits are allowed only on research pins named LN_<your-lane>_whatif_*, and those pins are not route evidence.\n";
    p += "Do not write a pin under an existing name. A symlink writes through into the main tree.\n";
    p += "Do not edit docs/STATUS.md. Do not edit BEAD_PLAN.md. Do not edit MASTER_PLAN.md unless you are the verdict manager.\n";
    p += "Do not run bd. Do not edit .beads/. Do not git add, git commit, git checkout, or git push. Do not revert unrelated dirty files.\n";
    p += "Re-read a source file immediately before you edit it. Keep uncommitted hunks that are already in the file. Soft max about 1000 lines: merge into the owning composer, do not add a sibling module.\n";
    p += "HEADLESS=1. No MP4. Glance is harvest.clock_glance. Natural entry is power-on. Do not start from Y1_D2_Morning_After_D1.\n";
    p += "ROM commands: cwd snes/harvest, SDL_VIDEODRIVER=dummy, QT_QPA_PLATFORM=offscreen, --trials 1, detached with setsid nohup. One emulator in your process. Do not kill other Python processes.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q. Docs tests: uv run pytest tests/test_docs.py -q.\n";
    p += "If the ROM contradicts the plan, stop. Write the contradiction into your research file and set blocked. Do not code around it.\n";
    p += "Two honest tries, then set blocked. Do not weaken a Done check.\n";
    return p;
}

fn mgr(agent, tasks, owns, extra) {
    let p = rules();
    p += "You are the manager " + agent + ".\n";
    p += "Do these beads yourself, in this order: " + tasks + ".\n";
    p += "You alone may create or edit: " + owns + ".\n";
    p += extra;
    p += "\nSkip a bead whose Done check is already true on disk. If a Deps bead is not done, stop that bead and put its id in blocked.\n";
    p += "Return completed (PR ids you finished), evidence (paths, commands, frame counts), and blocked (one id, or an empty string).\n";
    return p;
}

fn done_has(r, id) {
    if r == () || r.success != true || r.output == () || r.output.completed == () {
        return false;
    }
    if type_of(r.output.completed) != "array" {
        return false;
    }
    for c in r.output.completed {
        if c == id {
            return true;
        }
    }
    return false;
}

fn ids_of(r) {
    let s = "";
    if r == () || r.success != true || r.output == () || r.output.completed == () {
        return s;
    }
    if type_of(r.output.completed) != "array" {
        return s;
    }
    for c in r.output.completed {
        s += c;
        s += " ";
    }
    return s;
}

phase("File");
let filed = agent(
    "You are the beads manager. You are the only agent allowed to run bd or edit .beads/.\n"
    + "Repo: /home/v/01_projects/11_games/retro_rl.\n"
    + "Read snes/harvest/docs/perfect_rancher/BEAD_PLAN.md (How to file these, and every PR-001 through PR-100 entry) and docs/BEADS.md.\n"
    + "If `bd list -l rancher999` already prints issues, do not create duplicates. Return the existing count.\n"
    + "Otherwise file all 100 beads. Create PR-001 first. Create each epic with --parent set to the PR-001 id. Create each task and decision with --parent set to its epic id. PR-002 through PR-007 parent to PR-001.\n"
    + "Fields: -t epic|task|decision, -p from the P number (P0 is 0), -l harvest,rancher999,<kind>. Add the spine label only where the bead lists it.\n"
    + "--acceptance is the Done when line. Description starts with the plan key and the path snes/harvest/docs/perfect_rancher/BEAD_PLAN.md, then the Do line.\n"
    + "Add each Deps entry with `bd dep add <this> <dep>`. Existing rr- ids are real beads. A range such as PR-009..PR-018 means each of those beads.\n"
    + "Do not git add, commit, or push. Run `bd export -o .beads/issues.jsonl` when the graph is in.\n"
    + "Check: 100 issues with label rancher999, every dependency id exists, epics are 13, decisions are 3, tasks are 84.\n"
    + "Return filed (integer), root_id (the PR-001 bead id), and evidence (the count command output).",
    #{ label: "mgr-beads", capability_mode: "all", output_schema: file_schema }
);

let filed_n = 0;
if filed != () && filed.success == true && filed.output != () && filed.output.filed != () {
    filed_n = filed.output.filed;
}
log("filed " + filed_n.to_string());

phase("Wave 0");
let wave_jobs = [];
wave_jobs.push(#{
    prompt: mgr(
        "mgr-docs",
        "PR-002",
        "snes/harvest/docs/adr/0007-perfect-rancher.md, snes/harvest/CONTEXT.md, snes/harvest/docs/plan.md, the harvest manifest under docs/manifests/, and the generated game matrix",
        "Write ADR 0007: Perfect means RAM ranch_mastery 999 at ending 0x25, from one Clean continuous power-on run. Add the Perfect rung after Finish in CONTEXT.md. Rewrite the plan.md line that says 999 stays out. Set the harvest manifest ending contract, then run uv run python docs/generate_game_matrix.py from the repo root, then uv run pytest tests/test_docs.py -q. The owner asked for this campaign, so write that wording. Do not edit AGENTS.md."
    ),
    label: "mgr-docs",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-kit",
        "PR-006",
        "snes/harvest/scripts/pr999_lane.sh and a three-line trap note in snes/harvest/AGENTS.md",
        "Add snes/harvest/scripts/pr999_lane.sh <lane>. It creates worktree /home/v/01_projects/11_games/rr_harvest_pr999_<lane> on branch harvest/pr999-<lane>, symlinks roms, snes/harvest/.venv, the pin files, and the untracked task tapes, and adds an info/exclude entry for roms. Add a detached launcher (setsid nohup, then wait) and a pytest wrapper that sets PYTHONPATH to the worktree. Do not create the worktrees until the script exists, then create two lanes and run one pin probe plus the harvest no-ROM suite from each. Do not edit CONTEXT.md or plan.md."
    ),
    label: "mgr-kit",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-oracle",
        "PR-003, PR-007",
        "snes/harvest/harvest/core/ranch_master.py, snes/harvest/tests/test_ranch_master.py, research pins LN_oracle_whatif_*, and snes/harvest/docs/perfect_rancher/research/whatif_pins.md",
        "PR-003: the money term must read 24-bit money, not the low 16 bits. Replace invented rating titles and the Blue Feather text with ROM text ids. Unit-test the 9-bit wrap, the money wrap, all 12 cow slots, and the 999 cap. On the ROM, harvest.tools.ending_probe at least 6 states: LN_final_poweron_ending, the three showcase saves, and two mixed what-if pins. Oracle must equal RAM ranch_mastery after index 0x20. Then PR-007: build the LN_oracle_whatif_* pins listed in the bead with state_preset_builder. Mark them never-route-evidence. Do not edit the day planner."
    ),
    label: "mgr-oracle",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-calendar",
        "PR-021",
        "one new data module under snes/harvest/harvest/ for festival dates, shop hours, carpenter days, and girl locations, plus its test file",
        "Do not edit day_plan_orchestrator.py, multi_day_planner.py, ranch_master.py, run_to_day2.py, skills.py, or spring_opt.py. Sources are npc_catalog, the ROM, and live probes. A live probe must locate all 5 girls on 3 sample days."
    ),
    label: "mgr-calendar",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-happy",
        "PR-016",
        "snes/harvest/docs/perfect_rancher/research/F7_happiness.md and probe JSON next to it",
        "Research only. No product code. Decode AddPlayerHappiness callers and the 0x41 sites on $7F1F33. Rank repeatable positive sources and list sinks. Dated plan to reach happiness >= 992 by Y3 Summer D29."
    ),
    label: "mgr-happy",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-ship",
        "PR-017",
        "snes/harvest/docs/perfect_rancher/research/F8_shipping.md and probe JSON next to it",
        "Research only. No product code. Decode the $7F1F5E bit 0 gate around $818140. Probe whether a bin drop of N potatoes adds N, whether the count lands at drop or overnight, and whether the peddler, the saddlebag, or milk touch the four counters."
    ),
    label: "mgr-ship",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-hearts",
        "PR-018",
        "snes/harvest/docs/perfect_rancher/research/F9_hearts.md and probe JSON next to it",
        "Research only. No product code. For each of the 5 girls, list heart sources and sinks. Measure talk and gift deltas live on 2 girls. Measured deltas must match the decoded ones."
    ),
    label: "mgr-hearts",
    capability_mode: "all",
    output_schema: lane_schema,
});
wave_jobs.push(#{
    prompt: mgr(
        "mgr-gut",
        "PR-098",
        "snes/harvest/harvest/scripts/run_to_day2.py, snes/harvest/harvest/tasks/day_plan_orchestrator.py, snes/harvest/harvest/tasks/skills.py, and snes/harvest/harvest/spring_opt.py, plus tests that import only those modules",
        "Bring those four files under the soft max by merging into the existing composers. No sibling extracts. Fix the composer seams for social, family, and cover without adding new domain behavior. Keep every uncommitted hunk. Do not edit ranch_master.py. Run the harvest no-ROM suite. Start the power-on ending proof detached. If you can read a final frame count, completed requires it still equal 2175350. If the proof is still running when the suite is green, leave PR-098 out of completed, set blocked to PR-098, and put the log path in evidence."
    ),
    label: "mgr-gut",
    capability_mode: "all",
    output_schema: lane_schema,
});
let wave = parallel(wave_jobs);

let docs = ();
let kit = ();
let oracle = ();
let calendar = ();
let happy = ();
let ship = ();
let hearts = ();
let gut = ();
if wave != () && type_of(wave) == "array" && wave.len() >= 8 {
    docs = wave[0];
    kit = wave[1];
    oracle = wave[2];
    calendar = wave[3];
    happy = wave[4];
    ship = wave[5];
    hearts = wave[6];
    gut = wave[7];
}

let pins_ready = done_has(oracle, "PR-007");
let rom = [];
if pins_ready {
    phase("ROM gates");
    let rom_jobs = [];
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f0",
            "PR-009",
            "snes/harvest/docs/perfect_rancher/research/F0_drift.md and probe JSON next to it",
            "Research only. From LN_ending_Y3S29_eve and a near-perfect D29 what-if pin, sleep through to index 0x20. Record every score input before and after. Decode NightReset, the D29 6:00 event (0A/5), and CODE_83F4D8. Explain the Maria +2. Deliver a per-term drift model and a ledger projection formula."
        ),
        label: "mgr-f0",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f1",
            "PR-010",
            "snes/harvest/docs/perfect_rancher/research/F1_wife_hearts.md and probe JSON next to it",
            "Research only. Decode every negative heart site: gift, dialogue, festival, or event. Say whether it repeats, how often per day, and whether it still applies when married. Probe a married wife at 660 hearts down into 496-511. If it is impossible, write the proof and the natural maximum."
        ),
        label: "mgr-f1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f2",
            "PR-011",
            "snes/harvest/docs/perfect_rancher/research/F2_development.md and the layout draft next to it",
            "Research only. Tile census of a cleared farm. Which tiles can be grass, fence, or crop. Whether the farmer can walk on each counted type. Cost per tile. Whether cut grass stays at or above 0x1D. What hurricanes, season changes, and winter do. A layout with at most 24 uncovered tiles and paths to the house, bin, barn, coop, shed, and exit. Prove >= 2950 counted tiles."
        ),
        label: "mgr-f2",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f3",
            "PR-012",
            "snes/harvest/docs/perfect_rancher/research/F3_herd.md and probe JSON next to it",
            "Research only. Cow happiness byte at slot +4: buy value, birth value, talk, brush, milk, feed, outdoors, decay, sickness, death. Barn capacity, miracle potion, whether cow_N counts calves, coop capacity, incubation, chick growth. Latest date cow k reaches 200, from two probes: a bought cow on day 0, and one day of full care."
        ),
        label: "mgr-f3",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f4",
            "PR-013",
            "snes/harvest/docs/perfect_rancher/research/F4_family.md and probe JSON next to it",
            "Research only. Decode $7F1F66 wife bits, marriage requirements, wife_pregnancy (>= 20), gestation, child flags and ages in $7F1F6E, spacing between the two children, and the naming screens. Latest marriage date that still yields two children by Y3 Summer D29. Probe one pregnancy step from a what-if pin."
        ),
        label: "mgr-f4",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-f6",
            "PR-015",
            "snes/harvest/docs/perfect_rancher/research/F6_collectibles.md and probe JSON next to it",
            "Research only. Every power-berry source with trigger, dates, RNG, and count. Confirm the total is 10. Decode the clock ($7F1F6E & 0x4000) and the turtle shell ($7F1F6C & 0x1000). Note the duplication glitch and do not use it. Acquisition plan per item with earliest and latest dates."
        ),
        label: "mgr-f6",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom_jobs.push(#{
        prompt: mgr(
            "mgr-home",
            "PR-061",
            "snes/harvest/docs/perfect_rancher/research/carpenter.md and probe JSON next to it",
            "Research only. Upgrade flow: dialogs, cost, lumber, build days, flag bits $7F1F64 0x40 and 0x80. Probe one upgrade order from a funded what-if pin. No product code."
        ),
        label: "mgr-home",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    rom = parallel(rom_jobs);
} else {
    log("PR-007 did not finish; ROM gates stayed queued");
}

let f0 = ();
let f1 = ();
let f2 = ();
let f3 = ();
let f4 = ();
let f6 = ();
let home = ();
if rom != () && type_of(rom) == "array" && rom.len() >= 7 {
    f0 = rom[0];
    f1 = rom[1];
    f2 = rom[2];
    f3 = rom[3];
    f4 = rom[4];
    f6 = rom[5];
    home = rom[6];
}

let econ = ();
if done_has(f2, "PR-011") && done_has(f3, "PR-012") && done_has(f4, "PR-013") {
    phase("Economy");
    econ = agent(mgr(
        "mgr-f5",
        "PR-014",
        "snes/harvest/docs/perfect_rancher/research/F5_economy.md and a ledger output next to it. You may extend livestock_econ. Do not edit spring_opt.py if PR-098 is still open; write the 299-day ledger beside the research note instead.",
        "Price table from ROM tables and shop probes: house upgrades, brush, milker, sprinkler, grass seed, fodder, chicken feed, miracle potion, medicine, tomato seed, corn seed. Milk size against cow happiness, and egg output. The money path must end between 99840G and 655350G on D30 under the F2, F3, and F4 plans."
    ), #{ label: "mgr-f5", capability_mode: "all", output_schema: lane_schema });
} else {
    log("F5 stayed queued; F2, F3, or F4 is open");
}

let verdict = ();
let gates_ready = done_has(f0, "PR-009") && done_has(f1, "PR-010") && done_has(f2, "PR-011") && done_has(f3, "PR-012") && done_has(f4, "PR-013") && done_has(econ, "PR-014") && done_has(f6, "PR-015") && done_has(happy, "PR-016") && done_has(ship, "PR-017") && done_has(hearts, "PR-018");
if gates_ready {
    phase("Verdict");
    verdict = agent(mgr(
        "mgr-f10",
        "PR-019",
        "snes/harvest/docs/perfect_rancher/research/F10_verdict.md and MASTER_PLAN.md sections 3 and 4 only",
        "Combine F0 through F9. Set reachable true only if the notes show 999 with no glitch. Otherwise set reachable false, name the natural maximum, and name the short term. Name binding deadlines and any slack (chickens above 12, berries above 10). Do not decide the glitch, the lookahead rule, or the fallback. Those stay for the owner."
    ), #{ label: "mgr-f10", capability_mode: "all", output_schema: verdict_schema });
} else {
    log("F10 stayed queued; a gate note is missing");
}

let finished = "";
finished += ids_of(docs);
finished += ids_of(kit);
finished += ids_of(oracle);
finished += ids_of(calendar);
finished += ids_of(happy);
finished += ids_of(ship);
finished += ids_of(hearts);
finished += ids_of(gut);
finished += ids_of(f0);
finished += ids_of(f1);
finished += ids_of(f2);
finished += ids_of(f3);
finished += ids_of(f4);
finished += ids_of(f6);
finished += ids_of(home);
finished += ids_of(econ);
finished += ids_of(verdict);

phase("Beads");
let closer = agent(
    "You are the beads manager again. You are the only agent allowed to run bd or edit .beads/.\n"
    + "Repo: /home/v/01_projects/11_games/retro_rl. Do not git add, commit, or push.\n"
    + "Close a rancher999 bead only when its id is in this finished list and you can see the Done artifact on disk: "
    + finished
    + "\nFor every other PR id a manager marked blocked, add a bd comment with the reason and leave the bead open. Do not mark it blocked unless the research file says the ROM makes it impossible.\n"
    + "Run bd export -o .beads/issues.jsonl.\n"
    + "Return filed as the number of beads you closed, root_id as the PR-001 id, and evidence as the closed ids.",
    #{ label: "mgr-close", capability_mode: "all", output_schema: file_schema }
);

let reachable = false;
if verdict != () && verdict.success == true && verdict.output != () && verdict.output.reachable == true {
    reachable = true;
}

let wave1 = ();
if reachable && done_has(kit, "PR-006") {
    phase("Wave 1");
    let w1 = [];
    w1.push(#{
        prompt: mgr(
            "mgr-oracle-w1",
            "PR-004, PR-025",
            "the existing day-journal writer and ranch report block, plus a window-controller module merged into the oracle composer, and their tests",
            "Work in the main tree. PR-004: each morning, write per-term value, max, gap, and window_state into day_journal. Add the D30 projection from the F0 drift model. Extend the ending block with the oracle breakdown and ROM ranch_mastery at 0x20 and 0x25. A --power-on --until-day 3 JSON must have ledger rows, and the berry-only ending must still score 9. PR-025: a target lo,hi for a RAM counter with a hard overshoot guard, plus unit tests. If PR-032 and PR-072 are not done, leave PR-025 out of completed. Do not use the berry-duplication glitch."
        ),
        label: "mgr-oracle-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    w1.push(#{
        prompt: mgr(
            "mgr-campaign-w1",
            "PR-022, PR-026",
            "snes/harvest/docs/perfect_rancher/milestones.yaml, a solver module merged into the campaign composer, run_to_day2.py only for --save-season-pins, and their tests",
            "Use the lane worktree from pr999_lane.sh campaign if it exists, otherwise the main tree. Re-read run_to_day2.py before adding the flag. Do not edit day_plan_orchestrator.py or multi_day_planner.py. PR-022 needs F10 and F5 on disk. Solver tests must show a 10-day milestone move changing slack. PR-026: season pins PR_Y{n}_{Season}_D1 with a parent-run hash, and a resume that matches the next day frame for frame. Do not use the berry-duplication glitch."
        ),
        label: "mgr-campaign-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    w1.push(#{
        prompt: mgr(
            "mgr-farm-w1",
            "PR-029, PR-030, PR-031, PR-032, PR-033, PR-034, PR-035",
            "crop, shipping, shop, and money modules already under snes/harvest/harvest/, plus tests. No new sibling package.",
            "Use worktree harvest/pr999-farm via snes/harvest/scripts/pr999_lane.sh farm. Do a bead only when every Deps id is closed or the research file it needs is on disk. Do not edit day_plan_orchestrator.py, multi_day_planner.py, or run_to_day2.py. If a bead needs a splice into the day planner, leave it out of completed and say so. Do not use the berry-duplication glitch. Do not merge the worktree."
        ),
        label: "mgr-farm-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    w1.push(#{
        prompt: mgr(
            "mgr-animals-w1",
            "PR-037, PR-038, PR-039, PR-040, PR-041, PR-043, PR-044, PR-045, PR-046, PR-047, PR-049, PR-050",
            "the existing chicken and cow task modules under snes/harvest/harvest/, plus tests",
            "Use worktree harvest/pr999-animals via pr999_lane.sh animals. Skip PR-048 unless F3_herd.md says breeding is the path. Do a bead only when its Deps are done. Do not edit the day planner composers or run_to_day2.py. Do not merge the worktree. Do not use the berry-duplication glitch."
        ),
        label: "mgr-animals-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    w1.push(#{
        prompt: mgr(
            "mgr-social-w1",
            "PR-067, PR-068, PR-069, PR-070, PR-071, PR-072, PR-073, PR-074",
            "the existing social and festival modules under snes/harvest/harvest/, plus tests",
            "Use worktree harvest/pr999-social via pr999_lane.sh social. Marriage and children are later beads; do not start PR-076 or later. Do not edit the day planner composers. Do not merge the worktree. Do not use the berry-duplication glitch. Heart controllers must target 496-511, not a maximum."
        ),
        label: "mgr-social-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    w1.push(#{
        prompt: mgr(
            "mgr-home-w1",
            "PR-062, PR-065, PR-082, PR-083, PR-084, PR-085",
            "the existing tool, lumber, and collectible modules under snes/harvest/harvest/, plus tests",
            "Use worktree harvest/pr999-home via pr999_lane.sh home. Do a bead only when its Deps are done. House upgrade order is PR-063 and waits on lumber plus a natural pin; leave PR-063 out of this wave. Do not edit the day planner. Do not merge the worktree. Do not use the berry-duplication glitch."
        ),
        label: "mgr-home-w1",
        capability_mode: "all",
        output_schema: lane_schema,
    });
    wave1 = parallel(w1);
} else {
    log("Wave 1 stayed queued");
}

let w1_done = "";
if wave1 != () && type_of(wave1) == "array" {
    for slot in wave1 {
        w1_done += ids_of(slot);
    }
}

if w1_done != "" {
    agent(
        "You are the beads manager. Close rancher999 beads whose ids are in this list only when the Done artifact exists on disk: "
        + w1_done
        + "\nDo not close anything else. Do not git add, commit, or push. Run bd export -o .beads/issues.jsonl. Return filed as the number closed, root_id as the PR-001 id, and evidence as the closed ids.",
        #{ label: "mgr-close-w1", capability_mode: "all", output_schema: file_schema }
    );
}

let summary = "Perfect rancher managers finished a pass. Beads filed: " + filed_n.to_string() + ". Finished ids: " + finished + w1_done + ".";
if reachable {
    summary += " F10 says 999 is reachable without a glitch. Wave 1 ran in lane worktrees and did not merge.";
} else {
    summary += " F10 did not clear a no-glitch 999. Owner questions still open: is the power-berry duplication glitch allowed; do savestate rollouts stay dev-only with the final run replayed from power-on at 0 loads; if 999 is impossible, what is the fallback score. Wave 2 and later stay queued.";
}
let report_path = write_scratch_file("harvest-pr999.md", summary);
complete(#{ summary: summary, path: report_path });
