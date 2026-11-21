let meta = #{
    name: "harvest-herd-cash",
    description: "Fund the second cow by keeping the house and the broken plot off the wallet",
    when_to_use: "After harvest-herd-fund. Spring potatoes peaked at 2730G. Summer ended at 1670G.",
    phases: [
        #{ title: "Cash", detail: "defer the house and fix the second plot" },
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
    p += "No RAM writes. No --power-on. Do not start from LN_herd_name_ready, LN_herd_after_raise, or herd_cow1_f627470. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. home.py is about 382. day_phase_catalog.py is about 895. day_phase_registry.py and multi_day_planner.py and chicken_sale.py are at the cap. No new sibling module. Do not record a new walk.\n";
    p += "Controller input only. Chickens stay 0 while cows are under 2. The slot is created by the name script. Leave the farm-gate name hold in multi_nav.py alone.\n";
    return p;
}

phase("Cash");
let cash_body = rules();
cash_body += "You are the cash-path editor. No emulator in this phase.\n";
cash_body += "Cow 1 and Year 2 spring are already receipted. Do not replay them. The second cow failed in snes/harvest/recordings/herd_cow2.json. Start money 1690 on Y2 Spring D6. Peak 2730 on Spring D27 from potato shipments. Spring D29 bought 2 potato bags, 2730 to 2330. Summer D2 bought 1 corn bag, 2330 to 2030. Summer D15 bought 2 corn bags, 2270 to 1670. End Y2 Summer D24 wallet 1670, planted_grass 33, grass_seeds 0, cows 1, chickens 0, slots 1. Grass is enough. The cow price is 5000 plus a 300 reserve.\n";
cash_body += "HOUSE_WOOD began Spring D28 and ran most of early summer. apply_house_phases in harvest/planner/tasks/home.py arms the first house once eve_hearts is at least 250 outside Year 1 spring. This save crossed that line and spent the planting days chopping. While num_cows is under 2, the first house must not insert HOUSE_WOOD or HOUSE_UPGRADE. Married house-two gathering can stay. Update test_later_year_spring_can_still_order_the_first_house in tests/test_house_upgrade.py, which currently expects HOUSE_UPGRADE at num_cows 1. Do not weaken the married wood tests.\n";
cash_body += "The corn never reached the ground. prove.log around line 130054: CROP_ESTABLISH nav_pocket_plant timed out. The farmer was on the farm, the west pocket around (13, 28) already held potatoes, and the navigator push-faced tiles (18, 28), (18, 29), and (18, 32). Recovery position was (193, 470). The same timeout repeated Summer D14 through D20. Read POCKET_PLANT_CENTERS and farm_nav_pocket_hoe_stand_skill. Point the second pocket's hoe stand at a walkable tile the existing route can reach. Quote that tile's id. If you cannot name a walkable stand from the map data, passed is false and you stop. Do not record a walk.\n";
cash_body += "Run QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes/harvest/tests/test_house_upgrade.py snes/harvest/tests/test_crop_skills.py snes/harvest/tests/test_two_bag_replant.py -q --tb=line\n";
cash_body += "passed is true only if the first-house guard is in home.py, the second-pocket stand is a walkable tile you quote, and pytest shows no failures.\n";
let cash = agent(cash_body, #{ label: "cash", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(cash) {
    halt = "The cash path was not fixed.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover.\n";
    body += "Cash evidence: ";
    body += text_of(cash.output.evidence);
    body += "\nStart from --state LN_herd_y2_spring only. Ordinary planner. One run. Cap --until-year 2 --until-season 1 --until-day 24 --max-frames 800000. The CLI year flag is 1-based, so year 2 is ROM year byte 1. Do not run a second copy of this summer if the wallet is still under 5300. A crash before summer may be retried once from the same pin.\n";
    body += "Need cows 1 to 2, chickens 0, a new second slot, and a 5000 wallet drop. Grass must be at least 32 with zero chickens. Rain blocks delivery. Do not schedule a third cow. After payment, the farm-gate hold names the cow. Do not press A on the shop menu while 0x096E is 0x2E.\n";
    body += "Write snes/harvest/recordings/herd_cow2.json. Save LN_herd_after_cow2 only when the second slot exists. If summer D24 is still under 5300G, passed is false. Quote the wallet, grass, seeds, and whether HOUSE_WOOD ran.\n";
    body += "passed is true only if cows go from 1 to 2, chickens stay 0, a second slot exists, the wallet drops by 5000, ram_writes are 0, and mid_run_state_loads are 0.\n";
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
    body += "cow1 needs slot 0, happiness not lower after the house sleep, and ram_writes 0. spring needs year 1, season 0, chickens 0, and a potato tile other than 0xEB. cow2 needs cows at least 2, chickens 0, a second slot, and a 5000 wallet drop. accepted is true only when all three show those numbers and mid_run_state_loads 0. Quote the numbers.\n";
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
    if cash != () && cash.output != () {
        report += " Cash: ";
        report += text_of(cash.output.blocked);
        report += " ";
        report += text_of(cash.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
}
let path = write_scratch_file("herd_cash_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
