let meta = #{
    name: "harvest-pr999-f3",
    description: "Measure the fed-barn cow night, then the economy note and the F10 verdict",
    when_to_use: "After harvest-pr999-cont closed the calendar and left F3 without a date. One barn-night probe, then F5 and F10. Wave 1 only if 999 is reachable without a glitch.",
    phases: [
        #{ title: "Barn night", detail: "slot-0 trough and one sleep" },
        #{ title: "Economy", detail: "F5 after the measured night" },
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
    p += "Read snes/harvest/AGENTS.md. Read the bead sections named below in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Use read_file and grep. Do not answer from memory.\n";
    p += "Clean route code only. No LiveRamEditor and no --ram-set on a route. RAM edits are allowed only on a new research pin named LN_mgr-f3_whatif_barn0, and that pin is not route evidence.\n";
    p += "Do not write a pin under an existing name.\n";
    p += "Do not edit docs/STATUS.md or BEAD_PLAN.md. Do not edit MASTER_PLAN.md unless you are the verdict manager.\n";
    p += "Do not run bd. Do not edit .beads/. Do not git add, git commit, git checkout, or git push.\n";
    p += "Re-read a source file immediately before you edit it. Keep uncommitted hunks. Soft max about 1000 lines. No new sibling module.\n";
    p += "HEADLESS=1. No MP4. cwd snes/harvest. SDL_VIDEODRIVER=dummy. QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. Detach with setsid nohup.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q.\n";
    p += "Two honest tries, then set blocked. Do not weaken a Done check. Do not repeat a walk this prompt already says failed.\n";
    return p;
}

fn mgr(agent, tasks, owns, extra) {
    let p = rules();
    p += "You are the manager " + agent + ".\n";
    p += "Do these beads yourself, in this order: " + tasks + ".\n";
    p += "You alone may create or edit: " + owns + ".\n";
    p += extra;
    p += "\nReturn completed (PR ids you finished), evidence (paths, commands, frame counts), and blocked (one id, or an empty string).\n";
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

phase("Barn night");
let herd = agent(mgr(
    "mgr-f3",
    "PR-012",
    "snes/harvest/docs/perfect_rancher/research/F3_herd.md and one new probe JSON beside it. Do not delete F3_herd_day0.json, F3_herd_care3.json, or F3_herd_care4.json.",
    "Day-0 is already measured. research/F3_herd_day0.json frame 4803: bought cow happiness 0, status 0x05, map 0x27. Do not replay tasks/buy_cow.json and do not walk to the animal shop. The missing probe is one adult night in the barn with slot 0's own feed bit set. Read cow_geometry.py COW_FEED_SPOTS before you boot. Flag 0x0001 is stand (7, 17), face right, interact pixel exactly (113, 277). Flag 0x0008 is the other trough at (113, 149). Do not press A at (113, 275): F3_herd_care4.json dropped held fodder there and the flag stayed 0. Do not run CowChoresTask from (113, 277); F3_herd_care3.json hit max steps with fed flags still 0. The cow on buy_cow_end is map 39. A map of 39 takes the outdoors -8 even when a flag is set. On the new pin LN_mgr-f3_whatif_barn0, copy that adult, set slot +2 to 0x27, and do not write the happiness byte. Hold fodder, stand on (113, 277), face right, and press A the way cow_feed_ops does for the spot whose flag is 0x0001: face_frames 4, hold_frames 8, do not hold the face while A is down. Confirm fed flags include 0x0001 and the trough consumed the fodder. Then sleep from the house bed, not the barn. Record happiness before sleep and at 06:00. Append that night to F3_herd.md and write the latest date cow k reaches 200 from day-0 happiness 0, talk +1, brush +3, the decoded 21 child nights, and this measured adult night delta. If the first trough press still drops the fodder, read ToolUsedCowFeed at $82A4C2 and match the tile property $E2..$EE on the second try. Then stop."
), #{ label: "mgr-f3", capability_mode: "all", output_schema: lane_schema });

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
    log("F5 stayed queued; the barn night was not measured");
}

let verdict = ();
if done_has(herd, "PR-012") && done_has(econ, "PR-014") {
    phase("Verdict");
    verdict = agent(mgr(
        "mgr-f10",
        "PR-019",
        "snes/harvest/docs/perfect_rancher/research/F10_verdict.md and MASTER_PLAN.md sections 3 and 4 only",
        "Read F0_drift.md through F9_hearts.md. Set reachable true only if those notes show 999 with no glitch. F6 says the designed berry count is 7, and the best ending accepts power_berry_N only when it is exactly 10. Do not treat chicken deaths or the screen-transition edge as allowed. If the stamina term stays at 35, set reachable false, name the natural maximum, and name the short term. Do not amend ADR 0007. Do not decide the duplication edge, the savestate rule, or the fallback score."
    ), #{ label: "mgr-f10", capability_mode: "all", output_schema: verdict_schema });
} else {
    log("F10 stayed queued; F3 or F5 is open");
}

let finished = "";
finished += ids_of(herd);
finished += ids_of(econ);
finished += ids_of(verdict);

phase("Beads");
let closer = agent(
    "You are the beads manager. You are the only agent allowed to run bd or edit .beads/.\n"
    + "Repo: /home/v/01_projects/11_games/retro_rl. Do not git add, commit, or push.\n"
    + "Close a rancher999 bead only when its PR id is in this finished list and the Done artifact is on disk: "
    + finished
    + "\nId map: PR-012 is rr-ux6x.7.4, PR-014 is rr-ux6x.7.6, PR-019 is rr-ux6x.7.11. Do not close epics. Do not close PR-098.\n"
    + "If PR-012 is not in the finished list, add a bd comment on rr-ux6x.7.4 quoting the Latest date section of F3_herd.md, and leave the bead open.\n"
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
            "the existing day-journal writer, the ranch report block, and snes/harvest/harvest/core/ranch_master.py, plus their tests",
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
            "snes/harvest/docs/perfect_rancher/milestones.yaml, a solver merged into the campaign composer, run_to_day2.py only for --save-season-pins, and their tests",
            "Use the lane worktree from snes/harvest/scripts/pr999_lane.sh campaign if it exists, otherwise the main tree. Re-read run_to_day2.py before adding the flag. Do not edit day_plan_orchestrator.py or multi_day_planner.py. Solver tests must show a 10-day milestone move changing slack. PR-026: season pins with a parent-run hash, and a resume that matches the next day frame for frame. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-farm via snes/harvest/scripts/pr999_lane.sh farm. Do a bead only when its deps are done or the research file it needs is on disk. Do not edit the day planner composers or run_to_day2.py. If a bead needs a planner splice, leave it out of completed. Do not merge the worktree. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-animals via snes/harvest/scripts/pr999_lane.sh animals. Skip PR-048 unless F3_herd.md says breeding is the path. Do a bead only when its deps are done. Do not edit the day planner composers or run_to_day2.py. Do not merge the worktree. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-social via snes/harvest/scripts/pr999_lane.sh social. Do not start PR-076 or later. Do not edit the day planner composers. Do not merge the worktree. Heart controllers must target 496-511. Do not use the berry-duplication edge."
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
            "Use worktree harvest/pr999-home via snes/harvest/scripts/pr999_lane.sh home. Do a bead only when its deps are done. Leave PR-063 out. Do not edit the day planner. Do not merge the worktree. Do not use the berry-duplication edge."
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

let summary = "Barn-night continuation finished. Finished ids: " + finished + w1_done + ".";
if reachable {
    summary += " F10 says 999 is reachable without a glitch. Wave 1 ran in lane worktrees and did not merge.";
} else {
    summary += " F10 did not clear a no-glitch 999, or it did not run. Owner questions stay open. Wave 1 stays queued.";
}
let report_path = write_scratch_file("harvest-pr999-f3.md", summary);
complete(#{ summary: summary, path: report_path });
