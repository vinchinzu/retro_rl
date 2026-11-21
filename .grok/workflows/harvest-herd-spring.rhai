let meta = #{
    name: "harvest-herd-spring",
    description: "Main-flow first cow, raise it, buy a second cow, then plant Year 2 spring",
    when_to_use: "Continue the Harvest day planner until two cows are cared for and Year 2 spring plants. Stops when a receipt fails.",
    phases: [
        #{ title: "Audit", detail: "read the three gaps" },
        #{ title: "Cow1", detail: "buy and raise the first cow" },
        #{ title: "Cow2", detail: "buy the second cow" },
        #{ title: "Spring", detail: "Year 2 spring planting" },
        #{ title: "Verify", detail: "fail closed on the JSON" },
    ],
};

let audit_schema = #{
    "type": "object",
    "required": ["evidence", "cow1_gap", "cow2_gap", "spring_gap"],
    "properties": #{
        "evidence": #{ "type": "string" },
        "cow1_gap": #{ "type": "string" },
        "cow2_gap": #{ "type": "string" },
        "spring_gap": #{ "type": "string" },
        "predecessor": #{ "type": "string" },
        "spring_pin": #{ "type": "string" },
    },
};

let gate_schema = #{
    "type": "object",
    "required": ["passed", "evidence"],
    "properties": #{
        "passed": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "frames": #{ "type": "integer" },
        "report_path": #{ "type": "string" },
        "blocked": #{ "type": "string" },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["accepted", "cow1", "cow2", "spring", "evidence"],
    "properties": #{
        "accepted": #{ "type": "boolean" },
        "cow1": #{ "type": "boolean" },
        "cow2": #{ "type": "boolean" },
        "spring": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
    },
};

fn text_of(v) {
    if type_of(v) == "string" { return v; }
    if type_of(v) == "integer" { return v.to_string(); }
    return "";
}

fn gate_ok(r) {
    if r == () || r.success != true || r.output == () { return false; }
    if r.output.passed != true { return false; }
    if type_of(r.output.evidence) != "string" { return false; }
    if r.output.evidence == "" { return false; }
    return true;
}

fn rules() {
    let p = "";
    p += "Repo root: /home/v/01_projects/11_games/retro. Package harvest is snes/harvest. Import harvest.* from that tree.\n";
    p += "Read snes/harvest/AGENTS.md, snes/harvest/docs/plan.md, snes/harvest/docs/livestock_purchase_plan.md, and snes/harvest/docs/SPRING_ECONOMY.md before you edit or boot a ROM.\n";
    p += "This run is the ordinary day planner, build_day_phases_from_ram. It is not ranch-master 999 and not --perfect-rancher. Do not open perfect-rancher beads.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "Clean: controller input only. RAM writes forbidden. A mid-run state load does not count. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "Do not record a walk a search can already close. A shop menu, or a cross-map return to the origin, is not a buy.\n";
    p += "A cow buy needs the animal-shop tilemap, the wallet down by the cow price, num_cows up by 1, and a newly initialized slot after the name screen. Measured price is 5000G. The planner reserve stays 5000 plus 300.\n";
    p += "Animal buys stay in summer and fall. Do not open the shop in spring or winter. One animal per day. Rain and snow still block delivery.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. Glance with harvest.clock_glance.\n";
    p += "Command shape, from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 <flags>.\n";
    p += "Do not pass --day-plan buy_cow or --perfect-rancher. The default planner must schedule the work.\n";
    p += "Re-count a file before editing it. Soft max is about 1000 lines. multi_day_planner.py is at that cap: do not add lines to it. day_plan_phases.py and day_phase_catalog.py are near it: delete or replace lines in the file that already owns the behavior. No new sibling module.\n";
    p += "Cow purchase decisions live in harvest/planner/tasks/sale_menu.py. Pasture phases live in harvest/planner/day_phase_catalog.py pasture_day_phases. Outdoor and full-day assembly live in harvest/planner/day_plan_phases.py.\n";
    p += "Power-on still forces include_field_clear off. Do not flip that block.\n";
    p += "Two honest tries on a ROM receipt, then stop. Do not weaken a check. Do not mint a _vN report. Overwrite the one JSON path named in your task.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest <path> -q --tb=line. Mark new ROM tests with the existing rom marker. Do not boot a ROM from the no-ROM suite.\n";
    p += "A what-if pin whose date, gold, grass, hearts, or animal slots were written in RAM is not a predecessor. LN_*_whatif_* pins are not proof.\n";
    return p;
}

fn gate_prompt(role, body) {
    let p = rules();
    p += "You are ";
    p += role;
    p += ".\n";
    p += body;
    p += "\nReturn passed, evidence, frames, report_path, and blocked. passed is true only with a file path you opened and numbers you read from it. An empty evidence string is a failure.\n";
    return p;
}

phase("Audit");
let audit = agent(
    rules() + "You are the auditor. Read only. Use read_file and grep. Do not boot a ROM.\n"
        + "Confirm three gaps in the ordinary planner and quote the file and the condition.\n"
        + "cow1_gap: next_livestock buys a cow only when num_cows is 0, season is summer or fall, planted_grass is at least COW_PASTURE_TILES (32), and the wallet covers 5300G. Say whether any recording shows that buy from the default planner rather than a hand-run counter. The FixRainyDayEnd receipt recordings/rr-bu62.28-cow.json already had cows and is not main-flow proof.\n"
        + "Raised means the morning after that buy, the same planner runs cow chores for the new slot, feeds that slot's own bit when the cow is feedable, and one house-bed sleep does not drop that slot's happiness. A calf may skip the feed penalty. Do not require milk on day 0.\n"
        + "cow2_gap: next_livestock returns nothing for a cow once num_cows is not 0, and a funded cow no longer blocks the first chicken. The target of this run is two cows before the first chicken.\n"
        + "spring_gap: Year 2 spring is ROM year byte 1, season 0. Winter D1 wipes crops. Potato and turnip plant only in spring. Year 2 Spring D2 is an ordinary day, not the year-1 Spring D2 exhaustive tactic. Berry runs are already off when year is not 0. New Year on Year 2 Spring D1 is a forced festival.\n"
        + "predecessor: name a save under snes/harvest/custom_integrations/HarvestMoon-Snes/ that is a clean summer or fall morning with enough grass and gold for cow 1, or say none. Do not treat a RAM-edited pin as clean.\n"
        + "spring_pin: inspect LN_calendar_Y2_Spring_D1, LN_calendar_Y2_Spring_D2, and LN_ending_Y2S0D1_FAIL. Say which, if any, is a clean play successor rather than a projected or failed what-if. Quote how you decided.\n"
        + "Return evidence, cow1_gap, cow2_gap, spring_gap, predecessor, and spring_pin.",
    #{ label: "audit", capability_mode: "read-only", output_schema: audit_schema }
);

let halt = "";
let audit_json = "";
if audit == () || audit.success != true || audit.output == () || type_of(audit.output.evidence) != "string" || audit.output.evidence == "" {
    halt = "Audit returned no evidence.";
} else {
    audit_json = json_encode(audit.output);
    log("audit gaps recorded");
}

phase("Cow1");
let cow1 = ();
if halt == "" {
    let body = "";
    body += "Audit JSON follows. Check the code yourself before you trust it.\n";
    body += audit_json;
    body += "\nMake the ordinary planner buy the first cow and raise it for one night.\n";
    body += "Keep the first-cow gates: summer or fall, shop open, delivery weather clear, planted grass at least 32, wallet at least 5300G, num_cows is 0. Grass seed buys must still leave the cow fund. Do not buy a chicken on that morning.\n";
    body += "After the buy, the next morning's build_day_phases_from_ram list must include cow chores. Run those chores. Feed the new slot's own trough bit when that cow is feedable. Sleep from the house bed. Happiness on that slot must be greater than or equal to the value before that sleep. Baby and child nights may skip the feed penalty. Do not require milk on a calf.\n";
    body += "Prefer a clean predecessor the auditor named. If none exists, one power-on window may advance until the buy morning: --power-on --until-season 1 --until-day 25 --max-frames 900000 --save-season-pins --checkpoint-dir snes/harvest/logs/herd_cow1. A retry may load only the last clean checkpoint that run wrote, as its single start load, and the buy receipt itself must show mid_run_state_loads 0 and ram_writes 0.\n";
    body += "If the predecessor is already the buy morning, cap the window at --max-frames 120000 and include the following morning's chores and sleep.\n";
    body += "Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when the receipt passes.\n";
    body += "Update the no-ROM tests that encode the old first-cow-only schedule, then run snes/harvest/tests/test_animal_purchase.py and the day-plan tests you touched.\n";
    body += "passed is true only if you read frames greater than 0, num_cows 0 to 1, a new slot, wallet down, shop tilemap, the next morning's feed or calf-care evidence, happiness not down after the house sleep, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false. Put the blocker in blocked.\n";
    cow1 = agent(
        gate_prompt("the first-cow prover", body),
        #{ label: "cow1", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(cow1) {
        halt = "First cow buy and raise failed.";
    }
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = "";
    body += "Cow 1 evidence follows.\n";
    body += text_of(cow1.output.evidence);
    body += "\nExtend the ordinary planner so the second cow is bought before the first chicken.\n";
    body += "next_livestock returns cow when num_cows is 1, the shop gates in purchase_block_reason pass, and the wallet covers 5300G. While num_cows is under 2, do not return chicken if that cow can be paid for or is only waiting on grass. Do not schedule a third cow.\n";
    body += "The shop compares planted_grass with 16 times the animal total after the buy. Raise the pasture target so two cows and zero chickens still means 32 planted tiles, and so an already-owned chicken raises the target by 16. Keep the summer and fall day window. Do not RAM-poke grass or gold.\n";
    body += "Start from LN_herd_after_cow1 if cow 1 saved it and it is a clean successor. Otherwise use the clean predecessor already in the cow 1 JSON. Cap --max-frames 180000. The buy day may also run chores so cow 1 stays fed when feedable. One animal that day, the new cow.\n";
    body += "Write snes/harvest/recordings/herd_cow2.json. You may --save-end-state LN_herd_after_cow2 only when the receipt passes.\n";
    body += "Update no-ROM tests for the two-cow schedule and run those tests.\n";
    body += "passed is true only if you read frames greater than 0, num_cows 1 to 2, a second initialized slot, wallet down by the cow price, shop tilemap, cow 1 still present, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false. Do not revert the cow 1 planner path.\n";
    cow2 = agent(
        gate_prompt("the second-cow prover", body),
        #{ label: "cow2", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Spring");
let spring = ();
if halt == "" {
    let body = "";
    body += "Audit JSON follows.\n";
    body += audit_json;
    body += "\nCow evidence follows. Do not undo those planner edits. Do not change next_livestock, the cow price, or the pasture target.\n";
    if cow2 != () && cow2.output != () {
        body += text_of(cow2.output.evidence);
        body += "\n";
    }
    body += "Get the ordinary planner through the start of Year 2 spring planting.\n";
    body += "Year byte 1, season 0. Potato and turnip only. A tile id of 0xEB is a failed off-season plant and does not count. Year 2 Spring D2 must stay an ordinary day, not the year-1 D2 exhaustive tactic. Leave berry runs off when year is not 0. Leave include_field_clear off on power-on.\n";
    body += "New Year, Year 2 Spring D1, is a forced festival: FestivalEscapeTask must get the farmer off that map before a plant day. Do not spend the cow fund on seeds when num_cows is under 2 and the wallet is the cow reserve. This phase does not buy cows.\n";
    body += "In scope: festival escape, spring seed buy, hoe, establish, and water on the first open Year 2 spring mornings. Out of scope: marriage, the house, a third cow, berries, and the perfect score.\n";
    body += "Use a clean Year 2 spring pin if the auditor named one. LN_calendar_Y2_Spring_D1 and LN_calendar_Y2_Spring_D2 are candidates only after you confirm they were not date-poked. If none is clean, set passed false and name the missing predecessor. Do not edit the calendar to invent Year 2.\n";
    body += "Play from the morning after New Year through several plant days: --until-day 6 --max-frames 400000. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "Add or update a no-ROM test that a year-1 spring day list buys potato or turnip seeds and does not include the year-0 Spring D2 exhaustive clear or a berry phase. Run that test.\n";
    body += "passed is true only if you read frames greater than 0, year byte 1, season 0, at least one potato or turnip tile that is not 0xEB, a wallet or seed-stock change from the shop when a buy was required, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false.\n";
    spring = agent(
        gate_prompt("the year-2 spring planter", body),
        #{ label: "spring", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = "";
    body += "Read only. Open these files and no others for the verdict: snes/harvest/recordings/herd_cow1.json, snes/harvest/recordings/herd_cow2.json, snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "Cow 1 agent said: ";
    body += text_of(cow1.output.evidence);
    body += "\nCow 2 agent said: ";
    body += text_of(cow2.output.evidence);
    body += "\nSpring agent said: ";
    body += text_of(spring.output.evidence);
    body += "\nSet cow1 true only if herd_cow1.json shows frames greater than 0, cows ending at least 1, a new slot, wallet down, ram_writes 0, mid_run_state_loads 0, and happiness not down after a house sleep.\n";
    body += "Set cow2 true only if herd_cow2.json shows frames greater than 0, cows ending at least 2, wallet down from the start of that file, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Set spring true only if y2_spring_plant.json shows frames greater than 0, year 1, season 0, a potato or turnip tile other than 0xEB, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "accepted is true only when cow1, cow2, and spring are all true. Missing files are false. Quote the numbers you read.\n";
    verdict = agent(
        rules() + "You are the verifier.\n" + body,
        #{ label: "verify", capability_mode: "read-only", output_schema: verdict_schema }
    );
    if verdict == () || verdict.success != true || verdict.output == () || verdict.output.accepted != true {
        halt = "Verifier rejected the receipts.";
    }
}

let report = "";
if halt == "" && verdict != () && verdict.output != () {
    report += "Accepted. ";
    report += text_of(verdict.output.evidence);
} else {
    report += halt;
    if cow1 != () && cow1.output != () {
        report += " Cow1: ";
        report += text_of(cow1.output.blocked);
        report += " ";
        report += text_of(cow1.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
    if spring != () && spring.output != () {
        report += " Spring: ";
        report += text_of(spring.output.blocked);
        report += " ";
        report += text_of(spring.output.evidence);
    }
}
let path = write_scratch_file("herd_spring_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
