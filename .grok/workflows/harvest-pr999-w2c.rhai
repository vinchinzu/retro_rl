let meta = #{
    name: "harvest-pr999-w2c",
    description: "Storm recovery, the development census, and the campaign report",
    when_to_use: "After harvest-pr999-w2b closed PR-024 and PR-054. Do not repeat the hen walk, the lumber walk, or the girl locator.",
    phases: [
        #{ title: "Reports", detail: "storm what-if and the ledger report" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Read the named beads in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md and ADR 0007.\n";
    p += "Target stays 999. Berry count must land on exactly 10. Loads are allowed. RAM writes are forbidden on a route. A research pin named LN_<your-lane>_whatif_* may be edited and is not route evidence.\n";
    p += "Do not edit STATUS.md, BEAD_PLAN.md, MASTER_PLAN.md, ADR 0007, or milestones.yaml. Do not run bd. Do not git add, commit, or push. Do not merge a worktree.\n";
    p += "Re-read a file before editing it. Keep uncommitted hunks. No new sibling module except the report tool the bead names.\n";
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

phase("Reports");
let jobs = [];
jobs.push(#{
    prompt: mgr(
        "mgr-storm",
        "PR-057",
        "grass and layout modules in worktree /home/v/01_projects/11_games/rr_harvest_pr999_farm and their tests",
        "Detect counted tiles lost to a hurricane or a season change, and re-cover them. Done when a hurricane what-if recovers to the pre-storm count. F2 is the tile authority. The evening cover skill already planted 54 tiles from plan_new_crops_end. Do not edit the main-tree planner."
    ),
    label: "mgr-storm",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-report",
        "PR-058, PR-027, PR-086",
        "the day-journal writer on the main tree, snes/harvest/harvest/tools/pr999_report.py, and their tests",
        "PR-058: log counted tiles, the gap to 2950, and the linter's uncovered list in the journal of a checkpoint run. The layout linter is harvest/maps/farm_final_layout.py in the farm worktree; call it, do not edit it. PR-027: python -m harvest.tools.pr999_report prints term gaps, window states, and remaining collectible chances. It must run on a Y1 checkpoint JSON and flag a what-if lagging cow. PR-086: the same report lists the remaining berry chances on a checkpoint. The designed set is 7 and the eaten count must land on 10. Do not edit the farm worktree."
    ),
    label: "mgr-report",
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

let summary = "Storm and report pass finished. Finished ids: " + finished + ".";
let report_path = write_scratch_file("harvest-pr999-w2c.md", summary);
complete(#{ summary: summary, path: report_path });
