let meta = #{
    name: "harvest-pr999-w2",
    description: "Year planner reads the milestones, the farm layout linter, and flock growth to 4",
    when_to_use: "After harvest-pr999-w1b closed PR-025, PR-029, and PR-044. Do not repeat the girl locator, the lumber walk, or the animal-shop buy.",
    phases: [
        #{ title: "Next beads", detail: "planner, layout, flock" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Read the bead sections named below in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md and ADR 0007.\n";
    p += "Target stays 999. Berry edge may be used and must land on exactly 10. Loads are allowed. RAM writes are forbidden.\n";
    p += "Do not edit STATUS.md, BEAD_PLAN.md, MASTER_PLAN.md, or ADR 0007. Do not run bd. Do not git add, commit, or push. Do not merge a worktree.\n";
    p += "Re-read a file before editing it. Keep uncommitted hunks. No new sibling module. Soft max about 1000 lines.\n";
    p += "HEADLESS=1. SDL_VIDEODRIVER=dummy. QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Pytest: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q from the repo root.\n";
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

phase("Next beads");
let jobs = [];
jobs.push(#{
    prompt: mgr(
        "mgr-year",
        "PR-023",
        "the existing multi-day planner composer under snes/harvest/harvest/planner/ and its tests. Read milestones.yaml. Do not edit it.",
        "MultiDayPlannerTask picks day intents from snes/harvest/docs/perfect_rancher/milestones.yaml. Merge into the existing composer. No new runner. Done when a --power-on --until-day 5 journal shows intents taken from the milestones. Work on the main tree. Do not edit the farm worktree or the animals worktree."
    ),
    label: "mgr-year",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-layout",
        "PR-053",
        "a layout file and linter under snes/harvest/harvest/ in worktree /home/v/01_projects/11_games/rr_harvest_pr999_farm, plus their tests",
        "Target at least 2950 counted tiles, with paths to the house, bin, barn, coop, shed and exit. The linter diffs a live farm map into a per-tile to-do list. Done when the linter runs on 3 pins and the layout passes a BFS access check. F2 already measured 2968 walkable grass tiles. Do not edit multi_day_planner.py or the cow tasks."
    ),
    label: "mgr-layout",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-flock",
        "PR-037",
        "the chicken task modules in worktree /home/v/01_projects/11_games/rr_harvest_pr999_animals and their tests",
        "Buy the first chicken or take the free source, then incubate until the flock goes from 1 to at least 4. The growth rate has to match F3_herd.md. Done when a pin run shows that Clean, with 0 RAM writes. Do not walk to the animal shop by farm_to_animal_shop_counter. Do not start a cow at (113, 277). Do not edit the day planner or the farm layout files."
    ),
    label: "mgr-flock",
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

let summary = "Next-bead pass finished. Finished ids: " + finished + ".";
let report_path = write_scratch_file("harvest-pr999-w2.md", summary);
complete(#{ summary: summary, path: report_path });
