let meta = #{
    name: "harvest-pr999-w1",
    description: "Wave 1 of the Harvest perfect-rancher build after the owner allowed the berry edge",
    when_to_use: "After ADR 0007 records the 2026-09-30 ruling. Six file-disjoint lanes. The berry edge may be used and must land on exactly 10. Loads are allowed. The target stays 999.",
    phases: [
        #{ title: "Wave 1", detail: "six file-disjoint lanes" },
        #{ title: "Beads", detail: "close only finished ids" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. You are one Wave 1 manager on the Harvest perfect-rancher build.\n";
    p += "Read snes/harvest/docs/adr/0007-perfect-rancher.md, including the 2026-09-30 owner ruling, and the bead sections named below in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Use read_file. Do not answer from memory.\n";
    p += "The target is ranch_mastery 999 at ending 0x25. 984 is an acceptable finished score only if a run cannot land on 10 berries. Do not retarget the beads to 984.\n";
    p += "The power-berry duplication edge is allowed. power_berry_N must end at exactly 10. Any other count fails the best-ending check at $83F097. Do not pass 10.\n";
    p += "Savestate loads are allowed. RAM writes are forbidden. No LiveRamEditor and no --ram-set on a route.\n";
    p += "Do not edit docs/STATUS.md, BEAD_PLAN.md, MASTER_PLAN.md, or ADR 0007.\n";
    p += "Do not run bd. Do not edit .beads/. Do not git add, git commit, git checkout, or git push. Do not merge a worktree.\n";
    p += "Re-read a source file immediately before you edit it. Keep uncommitted hunks. Soft max about 1000 lines. No new sibling module.\n";
    p += "HEADLESS=1. No MP4. SDL_VIDEODRIVER=dummy. QT_QPA_PLATFORM=offscreen. One emulator in your process. Do not kill other Python processes.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q.\n";
    p += "Do a bead only when its deps are done or the research file it needs is on disk. If a bead needs a splice into a file you do not own, leave it out of completed and name the file.\n";
    p += "Two honest tries, then leave that bead out of completed. Do not weaken a Done check.\n";
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

phase("Wave 1");
let w1 = [];
w1.push(#{
    prompt: mgr(
        "mgr-oracle-w1",
        "PR-004, PR-025",
        "the existing day-journal writer, the ranch report block, and snes/harvest/harvest/core/ranch_master.py, plus their tests",
        "Work in the main tree. Do not edit livestock_econ.py, the day planner composers, or run_to_day2.py. PR-004: each morning, write per-term value, max, gap, and window_state into day_journal. Add the D30 projection from F0_drift.md. Extend the ending block with the oracle breakdown and ROM ranch_mastery at 0x20 and 0x25. A --power-on --until-day 3 JSON must have ledger rows, and the berry-only ending must still score 9. PR-025: a target lo,hi for a RAM counter with a hard overshoot guard, plus unit tests. The berry guard's high side is 10. If PR-032 and PR-072 are not done, leave PR-025 out of completed."
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
        "Use the lane worktree from snes/harvest/scripts/pr999_lane.sh campaign if it exists, otherwise the main tree. Re-read run_to_day2.py before adding the flag. Do not edit day_plan_orchestrator.py or multi_day_planner.py. Read F10_verdict.md and F5_economy.md. The berry milestone is exactly 10, using the allowed edge for the three beyond the designed seven. Solver tests must show a 10-day milestone move changing slack. PR-026: season pins with a parent-run hash, and a resume that matches the next day frame for frame. Loads are allowed."
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
        "Use worktree harvest/pr999-farm via snes/harvest/scripts/pr999_lane.sh farm. Do not edit the day planner composers, ranch_master.py, livestock_econ.py, or run_to_day2.py. If a bead needs a planner splice, leave it out of completed."
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
        "Use worktree harvest/pr999-animals via snes/harvest/scripts/pr999_lane.sh animals. Skip PR-048. F3_herd.md says slot 0 has no stall and a feedable slot reaches 200 on care day 50. A born calf can still fill slot 0. Do not edit the day planner composers, ranch_master.py, or run_to_day2.py."
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
        "Use worktree harvest/pr999-social via snes/harvest/scripts/pr999_lane.sh social. Do not start PR-076 or later. Do not edit the day planner composers. Heart controllers must target 496-511. The wife's hour-18 entry on the last night starts at 498-511 so the -2 lands in range."
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
        "Use worktree harvest/pr999-home via snes/harvest/scripts/pr999_lane.sh home. Do a bead only when its deps are done. Leave PR-063 out. Do not edit the day planner. The collectible beads may use the duplication edge. The eaten count has to be exactly 10, not 7 and not above 10."
    ),
    label: "mgr-home-w1",
    capability_mode: "all",
    output_schema: lane_schema,
});
let wave1 = parallel(w1);

let finished = "";
if wave1 != () && type_of(wave1) == "array" {
    for slot in wave1 {
        finished += ids_of(slot);
    }
}

phase("Beads");
agent(
    "You are the beads manager. You are the only agent allowed to run bd or edit .beads/.\n"
    + "Repo: /home/v/01_projects/11_games/retro_rl. Do not git add, commit, or push.\n"
    + "Close a rancher999 bead only when its PR id is in this finished list and the Done artifact is on disk: "
    + finished
    + "\nLook up each PR id in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Do not close epics. Do not close PR-098.\n"
    + "Run bd export -o .beads/issues.jsonl.\n"
    + "Return filed as the number of beads you closed, root_id as rr-ux6x, and evidence as the closed ids.",
    #{ label: "mgr-close-w1", capability_mode: "all", output_schema: file_schema }
);

let summary = "Wave 1 finished. Finished ids: " + finished + ". The owner ruling stands: the berry edge may land on exactly 10, loads are allowed, and the target stays 999.";
let report_path = write_scratch_file("harvest-pr999-w1.md", summary);
complete(#{ summary: summary, path: report_path });
