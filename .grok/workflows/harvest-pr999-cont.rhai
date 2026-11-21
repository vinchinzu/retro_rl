let meta = #{
    name: "harvest-pr999-cont",
    description: "Continue the Harvest perfect-rancher gates the first pass left open",
    when_to_use: "After harvest-pr999 filed rr-ux6x and closed the notes that were already true. Finishes F3, the calendar sightings, F5, and F10. Wave 1 runs only if F10 says 999 is reachable without a glitch.",
    phases: [
        #{ title: "Open gates", detail: "herd date and the five-girl sightings" },
        #{ title: "Frame delta", detail: "why the power-on ending is no longer 2175350" },
        #{ title: "Economy", detail: "F5 after a measured herd date" },
        #{ title: "Verdict", detail: "F10 from the notes on disk" },
        #{ title: "Beads", detail: "close only finished ids" },
        #{ title: "Wave 1", detail: "only if 999 is reachable without a glitch" },
    ],
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

let file_schema = #{
    "type": "object",
    "required": ["filed", "evidence"],
    "properties": #{
        "filed": #{ "type": "integer" },
        "evidence": #{ "type": "string" },
        "root_id": #{ "type": "string" },
    },
};

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. You are one manager on the Harvest perfect-rancher continuation.\n";
    p += "Read snes/harvest/AGENTS.md and snes/harvest/docs/perfect_rancher/MASTER_PLAN.md. Read the bead sections named below in BEAD_PLAN.md. Use read_file and grep. Do not answer from memory.\n";
    p += "The goal is ranch_mastery 999 at ending 0x25 from one Clean continuous power-on run.\n";
    p += "Clean route code only. No LiveRamEditor and no --ram-set on a route. RAM edits are allowed only on research pins named LN_<your-lane>_whatif_*, and those pins are not route evidence.\n";
    p += "Do not write a pin under an existing name. A symlink writes through into the main tree.\n";
    p += "Do not edit docs/STATUS.md, BEAD_PLAN.md, or MASTER_PLAN.md unless you are the verdict manager.\n";
    p += "Do not run bd. Do not edit .beads/. Do not git add, git commit, git checkout, or git push. Do not revert unrelated dirty files.\n";
    p += "Re-read a source file immediately before you edit it. Keep uncommitted hunks that are already in the file. Soft max about 1000 lines: merge into the owning composer, do not add a sibling module.\n";
    p += "HEADLESS=1. No MP4. Glance is harvest.clock_glance. Natural entry is power-on. Do not start from Y1_D2_Morning_After_D1.\n";
    p += "ROM commands: cwd snes/harvest, SDL_VIDEODRIVER=dummy, QT_QPA_PLATFORM=offscreen, --trials 1, detached with setsid nohup. One emulator in your process. Do not kill other Python processes.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q.\n";
    p += "If the ROM contradicts the plan, write the contradiction into your research file and set blocked. Do not code around it.\n";
    p += "Two honest tries, then set blocked. Do not weaken a Done check. Do not repeat a walk this prompt already says failed.\n";
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

phase("Open gates");
let open_jobs = [];
open_jobs.push(#{
    prompt: mgr(
        "mgr-f3",
        "PR-012",
        "snes/harvest/docs/perfect_rancher/research/F3_herd.md and new probe JSON files beside it. Do not delete F3_herd_probe.json.",
        "The decode in F3_herd.md stays. Do not rewrite it. The Done check is still open: a latest date for cow k reaching happiness 200, backed by a bought cow day-0 byte and one day of full care. Two failed facts you must not repeat: (1) farm_to_animal_shop_counter never created a slot. Try 1 died on tilemap 0x0C after push-facing tile (45, 26). Try 2 sat at (670, 834) at hour 18 with the shop closed and cows still empty. Do not start that walk again. (2) The care night on LN_mgr-f3_whatif_care2 went 44 to 36 because fed_cows_flags was 0x0008 and slot 0's night mask is 0x0001. Talk was +1, brush was +3, milk was 0. Sleeping in the barn idled. Do not use feed bit 0x0008 for slot 0, and do not sleep from the barn. Day-0: observe the slot byte on a live frame where SetCowNameBought has just run. A RAM-injected happiness is not that byte. An existing post-purchase recording or buy_cow_end may already contain the first frame the slot exists; read it before any write. If no such frame exists, do not walk the town. Full care: one adult already in a slot, feed that slot's own Cow_Feed_Flags bit, talk, brush, then sleep from the house bed. Record happiness before sleep and at 06:00. Weather bits 0x0002, 0x0008, 0x0010, and 0x0200 are the negative outdoors deltas. Then write the date from the measured day-0 byte, the measured night delta, and the already-decoded 21-night child growth. Two tries."
    ),
    label: "mgr-f3",
    capability_mode: "all",
    output_schema: lane_schema,
});
open_jobs.push(#{
    prompt: mgr(
        "mgr-calendar",
        "PR-021",
        "snes/harvest/harvest/calendar_schedule.py, snes/harvest/tests/test_calendar_schedule.py, and snes/harvest/tests/test_calendar_rom.py",
        "The data module already exists. recordings/pr021_calendar_probe.json has an empty sighting list on weekday, Saturday, and Sunday. The log recordings/pr021_calendar_probe.log shows npc_candidate sprites on tilemap 0x04 that were never named, and sprite_names stayed empty. The plaza route push-faced tile (46, 27) from (744, 424) and fell back to tilemap 0x0C. Do not repeat that east-gate walk. Name the sprites already in that log from npc_catalog. A girl whose sprite was on screen counts as a sighting. Then one new probe that locates all 5 girls on 3 sample days, using a pin that is already in town or already beside a girl. Done stays unmet if any of the 5 is missing on any sample day. Do not edit day_plan_orchestrator.py, multi_day_planner.py, ranch_master.py, run_to_day2.py, skills.py, or spring_opt.py."
    ),
    label: "mgr-calendar",
    capability_mode: "all",
    output_schema: lane_schema,
});
let opened = parallel(open_jobs);
let herd = ();
let calendar = ();
if opened != () && type_of(opened) == "array" && opened.len() >= 2 {
    herd = opened[0];
    calendar = opened[1];
}

phase("Frame delta");
let gut = agent(mgr(
    "mgr-gut",
    "PR-098",
    "snes/harvest/harvest/scripts/run_to_day2.py, snes/harvest/harvest/planner/day_plan_orchestrator.py, snes/harvest/harvest/planner/spring_opt.py, and snes/harvest/harvest/tasks/skills.py",
    "Those four files are already under 1000 lines. Do not split them again and do not add a sibling module. The no-ROM suite already passed. recordings/pr098_gut_power_on_ending.json reached credits at 3476992 frames with ram_writes 0 and mid_run_state_loads 0. logs/full_year/power_on_to_ending_farm.log reached the same credits at 2175350 frames. Diff the two journals and name the first day the frame delta diverges. If one seam in these four files added a stall, fix only that stall and run the harvest no-ROM suite. Do not start another --until-ending run. The Done check is still 2175350, so leave PR-098 out of completed. Set blocked to PR-098 and put the divergence day in evidence."
), #{ label: "mgr-gut", capability_mode: "all", output_schema: lane_schema });

let econ = ();
if done_has(herd, "PR-012") {
    phase("Economy");
    econ = agent(mgr(
        "mgr-f5",
        "PR-014",
        "snes/harvest/docs/perfect_rancher/research/F5_economy.md, a ledger file beside it, and snes/harvest/harvest/planner/livestock_econ.py",
        "Read F2_development.md, F3_herd.md, and F4_family.md before you build the ledger. Do not edit spring_opt.py, ranch_master.py, or the day planner composers. Price table from ROM tables and shop probes: house upgrades, brush, milker, sprinkler, grass seed, fodder, chicken feed, miracle potion, medicine, tomato seed, corn seed. Milk size against cow happiness, and egg output. The money path must end between 99840G and 655350G on D30 under those three plans. If F3 has no measured date, stop and set blocked to PR-012."
    ), #{ label: "mgr-f5", capability_mode: "all", output_schema: lane_schema });
} else {
    log("F5 stayed queued; F3 has no measured date");
}

let verdict = ();
if done_has(herd, "PR-012") && done_has(econ, "PR-014") {
    phase("Verdict");
    verdict = agent(mgr(
        "mgr-f10",
        "PR-019",
        "snes/harvest/docs/perfect_rancher/research/F10_verdict.md and MASTER_PLAN.md sections 3 and 4 only",
        "Read F0_drift.md, F1_wife_hearts.md, F2_development.md, F3_herd.md, F4_family.md, F5_economy.md, F6_collectibles.md, F7_happiness.md, F8_shipping.md, and F9_hearts.md. Combine them. Set reachable true only if those notes show 999 with no glitch. The designed berry count in F6 is 7, and the best ending accepts power_berry_N only when it is exactly 10. Chicken deaths and the screen-transition edge are the notes' extra three, and either can also pass 10. Do not treat either edge as allowed. If the notes leave the stamina term at 35, set reachable false, name the natural maximum, and name the short term. Name binding deadlines. Do not amend ADR 0007. Do not decide the duplication edge, the savestate rule, or the fallback score."
    ), #{ label: "mgr-f10", capability_mode: "all", output_schema: verdict_schema });
} else {
    log("F10 stayed queued; F3 or F5 is open");
}

let finished = "";
finished += ids_of(herd);
finished += ids_of(calendar);
finished += ids_of(gut);
finished += ids_of(econ);
finished += ids_of(verdict);

phase("Beads");
let closer = agent(
    "You are the beads manager. You are the only agent allowed to run bd or edit .beads/.\n"
    + "Repo: /home/v/01_projects/11_games/retro_rl. Do not git add, commit, or push.\n"
    + "Close a rancher999 bead only when its PR id is in this finished list and the Done artifact is on disk: "
    + finished
    + "\nId map: PR-012 is rr-ux6x.7.4, PR-014 is rr-ux6x.7.6, PR-019 is rr-ux6x.7.11, PR-021 is rr-ux6x.8.1, PR-098 is rr-ux6x.18.1. Look up any other PR id in BEAD_PLAN.md before closing it. Do not close epics.\n"
    + "PR-098 stays open even if it appears above, unless recordings show a power-on ending at exactly 2175350 frames. The known proof is 3476992 frames.\n"
    + "For a bead a manager left blocked, add a bd comment with the reason and leave it open.\n"
    + "Run bd export -o .beads/issues.jsonl.\n"
    + "Return filed as the number of beads you closed, root_id as rr-ux6x, and evidence as the closed ids.",
    #{ label: "mgr-close", capability_mode: "all", output_schema: file_schema }
);

let reachable = false;
if verdict != () && verdict.success == true && verdict.output != () && verdict.output.reachable == true {
    reachable = true;
}

let wave1 = ();
if reachable {
    phase("Wave 1");
    let w1 = [];
    w1.push(#{
        prompt: mgr(
            "mgr-oracle-w1",
            "PR-004, PR-025",
            "the existing day-journal writer and ranch report block, plus a window-controller module merged into snes/harvest/harvest/core/ranch_master.py, and their tests",
            "Work in the main tree. PR-004: each morning, write per-term value, max, gap, and window_state into day_journal. Add the D30 projection from F0_drift.md. Extend the ending block with the oracle breakdown and ROM ranch_mastery at 0x20 and 0x25. A --power-on --until-day 3 JSON must have ledger rows, and the berry-only ending must still score 9. PR-025: a target lo,hi for a RAM counter with a hard overshoot guard, plus unit tests. If PR-032 and PR-072 are not done, leave PR-025 out of completed. Do not use the berry-duplication edge."
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
            "Use the lane worktree from snes/harvest/scripts/pr999_lane.sh campaign if it exists, otherwise the main tree. Re-read run_to_day2.py before adding the flag. Do not edit day_plan_orchestrator.py or multi_day_planner.py. PR-022 needs F10_verdict.md and F5_economy.md on disk. Solver tests must show a 10-day milestone move changing slack. PR-026: season pins PR_Y{n}_{Season}_D1 with a parent-run hash, and a resume that matches the next day frame for frame. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-farm via snes/harvest/scripts/pr999_lane.sh farm. Do a bead only when every Deps id is closed or the research file it needs is on disk. Do not edit day_plan_orchestrator.py, multi_day_planner.py, or run_to_day2.py. If a bead needs a splice into the day planner, leave it out of completed and say so. Do not use the berry-duplication edge. Do not merge the worktree."
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
            "Use worktree harvest/pr999-animals via snes/harvest/scripts/pr999_lane.sh animals. Skip PR-048 unless F3_herd.md says breeding is the path. Do a bead only when its Deps are done. Do not edit the day planner composers or run_to_day2.py. Do not merge the worktree. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-social via snes/harvest/scripts/pr999_lane.sh social. Marriage and children are later beads; do not start PR-076 or later. Do not edit the day planner composers. Do not merge the worktree. Do not use the berry-duplication edge. Heart controllers must target 496-511, not a maximum."
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
            "Use worktree harvest/pr999-home via snes/harvest/scripts/pr999_lane.sh home. Do a bead only when its Deps are done. House upgrade order is PR-063 and waits on lumber plus a natural pin; leave PR-063 out. Do not edit the day planner. Do not merge the worktree. Do not use the berry-duplication edge."
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
        "You are the beads manager. Close rancher999 beads whose PR ids are in this list only when the Done artifact exists on disk: "
        + w1_done
        + "\nLook up each PR id in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Do not close epics. Do not git add, commit, or push. Run bd export -o .beads/issues.jsonl. Return filed as the number closed, root_id as rr-ux6x, and evidence as the closed ids.",
        #{ label: "mgr-close-w1", capability_mode: "all", output_schema: file_schema }
    );
}

let summary = "Perfect rancher continuation finished a pass. Finished ids: " + finished + w1_done + ".";
if reachable {
    summary += " F10 says 999 is reachable without a glitch. Wave 1 ran in lane worktrees and did not merge.";
} else {
    summary += " F10 did not clear a no-glitch 999, or it did not run. Owner questions stay open: is the power-berry duplication edge allowed when the designed set is 7 and the best ending wants exactly 10; do savestate rollouts stay dev-only with the final run replayed from power-on at 0 loads; if 10 berries cannot be reached without passing 10, is 984 the fallback. Wave 1 and later stay queued.";
}
let report_path = write_scratch_file("harvest-pr999-cont.md", summary);
complete(#{ summary: summary, path: report_path });
