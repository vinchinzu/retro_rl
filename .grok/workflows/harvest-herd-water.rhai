let meta = #{
    name: "harvest-herd-water",
    description: "Water the corn through harvest so the second cow fund can clear",
    when_to_use: "After harvest-herd-cash. Summer peaked at 2810G because the corn was never shipped.",
    phases: [
        #{ title: "Water", detail: "do not accept a water stand 33px away" },
        #{ title: "Cow2", detail: "one summer run for the second cow" },
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
    p += "No RAM writes. No --power-on. Do not start from LN_herd_name_ready or LN_herd_after_raise. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. navigation.py and the crop water task may be edited. day_phase_registry.py, multi_day_planner.py, and chicken_sale.py are at the cap. No new sibling module. Do not record a new walk.\n";
    p += "Controller input only. Chickens stay 0 while cows are under 2. Leave the first-house guard in home.py and the farm-gate name hold in multi_nav.py alone.\n";
    return p;
}

phase("Water");
let water_body = rules();
water_body += "You are the water editor. No emulator in this phase.\n";
water_body += "The latest herd_cow2.json is the summer that peaked at 2810G on Summer D7 and ended Summer D24 at 2170G. planted_grass ended at 31. grass_seeds ended at 1. HOUSE_WOOD did not run. Cows stayed 1. A potato plot cannot fund the cow: sell price 80, seed 200, one bag is 8 tiles, and six perfect cycles from Spring D6 still finish under 5300. Corn can. It costs 300, sells for 120, first harvest in 10 days, then every 3. Eight tiles shipped three times is 2880G. Added to the 2670G wallet on Summer D8, that clears 5300.\n";
water_body += "Summer D8 bought one corn bag, 2810 to 2510, and the plot was watered through about Summer D16. NAV_CROP then returned soft arrived dist=33 stasis=90. Summer D19 through D23 failed with water fail: dry_crops=7 watered=0 skipped=7. No corn shipment is in the journal. Read NavSkill._soft_arrived in harvest/planner/tasks/navigation.py. dist=33 must not count as arrived for a watering stand. The can has to be on a tile that can reach the crop. If the farmer is stuck farther away, fail the nav so it can repath. Do not loosen other hops. Quote the new water distance.\n";
water_body += "Summer D20 bought one grass bag, 2670 to 2170, and never planted it. The cow gate is 32 planted tiles. 31 does not qualify. The existing grass planter has to sow that bag when planted_grass is under 32 and a seed is in the bag. Do not add a new route.\n";
water_body += "Add or update a unit test that rejects a 33px water stand. Run QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes/harvest/tests/test_crop_skills.py snes/harvest/tests/test_house_upgrade.py -q --tb=line\n";
water_body += "passed is true only if you read the edited distance check, a test rejects dist 33, and pytest shows no failures.\n";
let water = agent(water_body, #{ label: "water", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(water) {
    halt = "The water stand was not fixed.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover.\n";
    body += "Water evidence: ";
    body += text_of(water.output.evidence);
    body += "\nStart from --state LN_herd_y2_spring only. Ordinary planner. One run. Cap --until-year 2 --until-season 1 --until-day 24 --max-frames 800000. The CLI year flag is 1-based. Do not run this summer again if the wallet is still under 5300. A crash before Summer D8 may be retried once from the same pin.\n";
    body += "Corn has to be shipped. A potato-only wallet under 5300 is a failure even if the days complete. Grass must end at 32 or more. Chickens stay 0. After the 5000 drop, the farm-gate hold names the cow. Do not press A on the shop menu while 0x096E is 0x2E.\n";
    body += "Write snes/harvest/recordings/herd_cow2.json. Save LN_herd_after_cow2 only when the second slot exists. passed is true only if cows go from 1 to 2, chickens stay 0, a second slot exists, the wallet drops by 5000, planted grass is at least 32, ram_writes are 0, and mid_run_state_loads are 0. If summer D24 is still under 5300, quote the wallet, the corn shipments, grass, and seeds.\n";
    cow2 = agent(body, #{ label: "cow2", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = rules();
    body += "You are the verifier. Read only. Open snes/harvest/recordings/herd_cow1.json, y2_spring_plant.json, and herd_cow2.json.\n";
    body += "cow1 needs slot 0 and ram_writes 0. spring needs year 1, season 0, and a potato tile other than 0xEB. cow2 needs cows at least 2, chickens 0, a second slot, grass at least 32, and a 5000 wallet drop. accepted is true only when all three show those numbers and mid_run_state_loads 0. Quote the numbers.\n";
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
    if water != () && water.output != () {
        report += " Water: ";
        report += text_of(water.output.blocked);
        report += " ";
        report += text_of(water.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
}
let path = write_scratch_file("herd_water_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
