let meta = #{
    name: "harvest-pr999-w2b",
    description: "Daily arbiter, then grass cover at scale",
    when_to_use: "After harvest-pr999-w2 closed PR-023 and PR-053. Do not repeat the town hen walk.",
    phases: [
        #{ title: "Unlocked", detail: "arbiter and grass cover" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Read the named bead in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md and ADR 0007.\n";
    p += "Target stays 999. Berry count must land on exactly 10. Loads are allowed. RAM writes are forbidden.\n";
    p += "Do not edit STATUS.md, BEAD_PLAN.md, MASTER_PLAN.md, ADR 0007, or milestones.yaml. Do not run bd. Do not git add, commit, or push. Do not merge a worktree.\n";
    p += "Re-read a file before editing it. Keep uncommitted hunks. No new sibling module.\n";
    p += "HEADLESS=1. SDL_VIDEODRIVER=dummy. QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q.\n";
    p += "Two honest tries, then leave the bead out of completed. Do not weaken a Done check.\n";
    return p;
}

fn mgr(agent, tasks, owns, extra) {
    let p = rules();
    p += "You are " + agent + ". Beads: " + tasks + ".\n";
    p += "You alone may edit: " + owns + ".\n";
    p += extra;
    p += "\nReturn completed, evidence, and blocked.\n";
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

phase("Unlocked");
let jobs = [];
jobs.push(#{
    prompt: mgr(
        "mgr-arbiter",
        "PR-024",
        "the existing multi-day planner composer under snes/harvest/harvest/planner/ and its tests, on the main tree",
        "Order the day's blocks: deadline-critical, then animals, then crops, then social, then cover. Give every block a frame account. Add the post-18:00 lane for cover and hoeing, with hot-spring refills. Merge into the existing composer. Done when the arbitration unit tests pass and a full-scale what-if day runs every block with per-block frames in the journal. Do not edit the farm worktree."
    ),
    label: "mgr-arbiter",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-grass",
        "PR-054",
        "grass and crop modules in worktree /home/v/01_projects/11_games/rr_harvest_pr999_farm and their tests",
        "PR-053's layout is harvest/maps/farm_final_layout.py in that worktree: 2968 grass tiles, BFS access from (8, 26). Read F2_development.md for the per-evening tile target N. Till and seed in the post-18:00 lane, with hot-spring stamina refills. Done when one pin evening covers at least N tiles and the log has tiles per frame. If the skill has to be spliced into multi_day_planner.py, leave PR-054 out of completed and name that file. Do not edit the main-tree planner."
    ),
    label: "mgr-grass",
    capability_mode: "all",
    output_schema: lane_schema,
});
let opened = parallel(jobs);

let finished = "";
if opened != () && type_of(opened) == "array" {
    for slot in opened {
        finished += ids_of(slot);
    }
}

phase("Beads");
agent(
    "You are the beads manager. Close a rancher999 bead only when its PR id is in this list and the Done artifact is on disk: "
    + finished
    + "\nLook up each id in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Do not close epics. Do not git add, commit, or push. From the repo root run bd export -o .beads/issues.jsonl. Return filed, root_id rr-ux6x, and evidence.",
    #{ label: "mgr-close", capability_mode: "all", output_schema: file_schema }
);

let summary = "Arbiter and grass pass finished. Finished ids: " + finished + ".";
let report_path = write_scratch_file("harvest-pr999-w2b.md", summary);
complete(#{ summary: summary, path: report_path });
