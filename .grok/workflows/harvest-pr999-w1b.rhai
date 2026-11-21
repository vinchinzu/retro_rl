let meta = #{
    name: "harvest-pr999-w1b",
    description: "Wire the window guard, put crop quotas in the milestones, and milk from a real cow stand",
    when_to_use: "After harvest-pr999-w1. PR-025, PR-029, and PR-044 only. Do not repeat the girl locator or the lumber walk.",
    phases: [
        #{ title: "Remainder", detail: "window guard, quotas, one milk day" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Read snes/harvest/docs/adr/0007-perfect-rancher.md and the bead sections named below in BEAD_PLAN.md.\n";
    p += "Target stays 999. The berry edge may be used and must land on exactly 10. Loads are allowed. RAM writes are forbidden.\n";
    p += "Do not edit docs/STATUS.md, BEAD_PLAN.md, MASTER_PLAN.md, or ADR 0007. Do not run bd. Do not git add, commit, or push. Do not merge a worktree.\n";
    p += "Re-read a file before editing it. Keep uncommitted hunks. No new sibling module.\n";
    p += "HEADLESS=1. SDL_VIDEODRIVER=dummy. QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen uv run pytest <path> -q. Lane tests go through snes/harvest/scripts/pr999_lane.sh.\n";
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

phase("Remainder");
let jobs = [];
jobs.push(#{
    prompt: mgr(
        "mgr-window",
        "PR-025",
        "snes/harvest/harvest/core/ranch_master.py, the potato window controller in worktree /home/v/01_projects/11_games/rr_harvest_pr999_farm, and the heart controller in worktree /home/v/01_projects/11_games/rr_harvest_pr999_social, plus the tests those controllers already have",
        "WindowGuard and berry_window_guard already exist and the synthetic tests already pass. PR-032 and PR-072 are closed and do not call WindowGuard. Add that call inside the existing potato shipper and the existing heart controller. Do not add a new module. social.py is already over 1000 lines, so the call is a few lines in the function that already moves hearts. The berry guard's high side stays 10. Done when the unit tests pass and both controllers import WindowGuard. Do not edit milestones.yaml or the cow tasks."
    ),
    label: "mgr-window",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-quota",
        "PR-029",
        "snes/harvest/docs/perfect_rancher/milestones.yaml and snes/harvest/tests/test_campaign_composer.py",
        "The quota numbers already live as CROP_QUOTA_MILESTONES in the farm worktree's crop planner, and the overshoot tests there are green. Copy those targets into milestones.yaml: potato and turnip close by Y3 Spring D30, tomato and corn by Y3 Summer D29, each at 496-511. Extend the campaign composer test so an overshoot target is rejected. Do not edit the farm worktree, spring_opt.py, or ranch_master.py."
    ),
    label: "mgr-quota",
    capability_mode: "all",
    output_schema: lane_schema,
});
jobs.push(#{
    prompt: mgr(
        "mgr-milk",
        "PR-044",
        "the cow task modules in worktree /home/v/01_projects/11_games/rr_harvest_pr999_animals and their tests",
        "Two boots already failed. Both started at (113, 277) and stalled in MILK_NAV walking toward (163, 278), with fed flags ending at 0x0008. Do not start at (113, 277) and do not press A there. Read cow_geometry.py for the brush and milk stands, and cow_interact_pixel. The feed stall for slot 1 is (113, 245) facing tile (8, 15), property 0xE3, flag 0x0002. One pin day must brush and milk one cow, ship the milk, and record the happiness delta. Use the animals worktree. Two tries, then stop."
    ),
    label: "mgr-milk",
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
    + "\nLook up each id in snes/harvest/docs/perfect_rancher/BEAD_PLAN.md. Do not close epics. Do not git add, commit, or push. Run bd export -o .beads/issues.jsonl from the repo root. Return filed, root_id rr-ux6x, and evidence.",
    #{ label: "mgr-close", capability_mode: "all", output_schema: file_schema }
);

let summary = "Wave 1 remainder finished. Finished ids: " + finished + ".";
let report_path = write_scratch_file("harvest-pr999-w1b.md", summary);
complete(#{ summary: summary, path: report_path });
