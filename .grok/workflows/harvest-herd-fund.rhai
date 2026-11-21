let meta = #{
    name: "harvest-herd-fund",
    description: "Keep the second cow ahead of the chicken, then plant Year 2 spring",
    when_to_use: "After harvest-herd-exit. Cow 1 is named and slept. Fall ended at 730G with a chicken.",
    phases: [
        #{ title: "Hold", detail: "refuse the chicken while cows are under 2" },
        #{ title: "Spring", detail: "Year 2 spring potato or turnip" },
        #{ title: "Cow2", detail: "second cow in Year 2 summer" },
        #{ title: "Verify", detail: "fail closed on the JSON" },
    ],
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
    p += "Repo root: /home/v/01_projects/11_games/retro. Package harvest is snes/harvest.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "No RAM writes. No --power-on. Do not start from herd_cow1_f627470, LN_herd_shop_receipt, or LN_herd_name_ready. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. sale_menu.py is about 468 and is the livestock file. multi_day_planner.py and day_phase_registry.py are at the cap. chicken_sale.py is at the cap. No new sibling module.\n";
    p += "Controller input only. Count plus a wallet drop is not a cow. The slot is created by the name script. multi_nav already holds the farm-gate direction for 90 frames while 0x7F1F5A bit 0x0002 is set. Do not remove that hold.\n";
    return p;
}

phase("Hold");
let hold_body = rules();
hold_body += "You are the livestock editor. No emulator.\n";
hold_body += "Read next_livestock in snes/harvest/harvest/planner/tasks/sale_menu.py before editing. Below 5300G it returns chicken unless HARVEST_HOLD_CHICKEN_FOR_COW is 1. The Fall D29 run did not set that variable. Wallet 1580 became 580, num_chickens became 1, num_cows stayed 1. Peak wallet on that run was 1580G, so the cow was not affordable. The chicken is still the wrong buy while cows are under 2.\n";
hold_body += "Change next_livestock so num_cows under 2 never returns chicken. Do not hide the rule behind an environment variable. A funded cow that is only waiting on grass still returns None. A ready cow still returns cow. Spring and winter still return None. Update the docstring.\n";
hold_body += "Find the existing test that expects a chicken while cows are under 2, and change that expectation. If no test covers it, add the assertion in the nearest existing livestock test file. Run: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes/harvest/tests/test_seasonal_prep.py snes/harvest/tests/test_animal_purchase.py snes/harvest/tests/test_day_plan_coop.py -q --tb=line\n";
hold_body += "passed is true only if you read the edited function and the pytest summary shows no failures. Quote the new return for a summer day, 1 cow, 0 chickens, 2000G, and enough grass.\n";
let hold = agent(hold_body, #{ label: "hold", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(hold) {
    halt = "The chicken hold did not land.";
}

phase("Spring");
let spring = ();
if halt == "" {
    let body = rules();
    body += "You are the year-2 spring planter.\n";
    body += "Hold evidence: ";
    body += text_of(hold.output.evidence);
    body += "\nCow 1 is already named and slept. herd_cow1.json records slot 0, Fall D21 to D22, happiness 0 then 1 then 1, ram_writes 0. Do not name that cow again.\n";
    body += "Start from --state LN_herd_after_raise. Measured: Year 1 Fall D22 06:00, house tilemap 21, money 530, num_cows 1, chickens 0, planted_grass 33, grass_seeds 0. Winter D1 wipes that grass. Animal buys are summer and fall only. This fall cannot reach 5300G. The last run peaked at 1580G and bought a chicken. Do not replay those 9 days as a cow attempt.\n";
    body += "Ordinary planner. Potato or turnip only. Tile 0xEB does not count. New Year Day 1 is FestivalEscapeTask. Year 2 Spring D2 is an ordinary day. Berries stay off when the year byte is not 0. Do not flip include_field_clear. Chickens must stay 0.\n";
    body += "Cap --until-year 1 --until-season 0 --until-day 6 --max-frames 500000. If that cap ends before Year 2 Spring D1, one second run at 800000 from the same pin. A mid-run state load does not count. Write snes/harvest/recordings/y2_spring_plant.json. Save LN_herd_y2_spring only when a real crop tile exists.\n";
    body += "passed is true only if frames are greater than 0, year is 1, season is 0, chickens stay 0, at least one potato or turnip tile exists and is not 0xEB, ram_writes are 0, and mid_run_state_loads are 0.\n";
    spring = agent(body, #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover.\n";
    body += "Spring evidence: ";
    body += text_of(spring.output.evidence);
    body += "\nStart from LN_herd_y2_spring. Earn the second cow with the ordinary planner. The next animal window is Year 2 summer. Winter wiped the 33 grass, so the pasture has to be planted again. Shop grass is 16 tiles per resulting animal. The cow also waits for 32 planted grass and 5300G. One animal that day. Rain blocks delivery. Do not schedule a third cow. Chickens must stay 0.\n";
    body += "Payment is not the slot. After the wallet drops, the name bit is 0x7F1F5A bit 0x0002. The farm-gate hold in multi_nav is what lets 0x096E become 0x42. Then the name screen creates the slot. Do not press A on the shop menu while 0x096E is 0x2E.\n";
    body += "Cap --until-year 1 --until-season 1 --until-day 24 --max-frames 900000. If summer ends under 5300G, set passed false and quote the wallet, grass, and seeds. Two tries only when the first dies before summer. Write snes/harvest/recordings/herd_cow2.json. Save LN_herd_after_cow2 only when the second slot exists.\n";
    body += "passed is true only if cows go from 1 to 2, chickens stay 0, a second new slot exists, the wallet drops by 5000, ram_writes are 0, and mid_run_state_loads are 0.\n";
    cow2 = agent(body, #{ label: "cow2", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = rules();
    body += "You are the verifier. Read only. Open snes/harvest/recordings/herd_cow1.json, herd_cow2.json, and y2_spring_plant.json.\n";
    body += "cow1 needs slot 0, happiness not lower after the house sleep, and ram_writes 0. cow2 needs cows at least 2, chickens 0, a second slot, and a 5000 wallet drop. spring needs year 1, season 0, chickens 0, and a potato or turnip tile other than 0xEB. accepted is true only when all three files show those numbers and mid_run_state_loads 0. Quote the numbers.\n";
    verdict = agent(body, #{ label: "verify", capability_mode: "read-only", output_schema: verdict_schema });
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
    if hold != () && hold.output != () {
        report += " Hold: ";
        report += text_of(hold.output.blocked);
        report += " ";
        report += text_of(hold.output.evidence);
    }
    if spring != () && spring.output != () {
        report += " Spring: ";
        report += text_of(spring.output.blocked);
        report += " ";
        report += text_of(spring.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
}
let path = write_scratch_file("herd_fund_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
