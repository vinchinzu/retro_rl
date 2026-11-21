let meta = #{
    name: "harvest-herd-ripe",
    description: "Ship the skipped corn ear and leave the barn, then buy the second cow",
    when_to_use: "After harvest-herd-water. Summer D24 ended at 4230G with corn tile (12, 28) unshipped.",
    phases: [
        #{ title: "Ripe", detail: "harvest stand, barn door, water repath" },
        #{ title: "Cow2", detail: "one summer through day 30" },
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
    p += "Repo root: /home/v/01_projects/11_games/retro. Package harvest is snes/harvest. Run pytest and the game from the repo root through uv run.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "No RAM writes. No --power-on. Do not start from LN_herd_name_ready, LN_herd_shop_receipt, LN_herd_after_raise, or herd_cow1_f627470. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. harvest_task.py is 926 lines, navigation.py is 665, transitions.py is 727. day_phase_registry.py, multi_day_planner.py, and chicken_sale.py are at the cap. Do not add lines to those three. No new sibling module. Do not record a new walk.\n";
    p += "Controller input only. Chickens stay 0 while cows are under 2. Leave the first-house guard in home.py and the farm-gate name hold in multi_nav.py alone. Leave WATER_STAND_ARRIVE_PX at 23. Do not turn include_field_clear on.\n";
    return p;
}

phase("Ripe");
let ripe_body = rules();
ripe_body += "You are the ripe-crop editor. No emulator in this phase.\n";
ripe_body += "Cow 1 is already in recordings/herd_cow1.json. Year 2 spring is already in recordings/y2_spring_plant.json. Do not redo either run.\n";
ripe_body += "Open recordings/herd_cow2.json and logs/herd_cow2/cow2_summer.log. The run started at LN_herd_y2_spring with 1690G and ended Y2 Summer D24 06:00 at 4230G. Corn shipments were 12 (D19 shipped 5, D20 shipped 2, D23 shipped 5). Wallet math is closed: 30 potato shipments at 80G plus 12 corn at 120G, minus five potato bags and one corn bag. Grass ended at 33. Chickens stayed 0. Cows stayed 1. The plants were still being watered on D23 (watered=7). The run stopped before Summer D30, so a later flush was never played. Do not treat 4230G as a dead crop.\n";
ripe_body += "Three defects, all named in that log. Fix all three.\n";
ripe_body += "1. Harvest tile (12, 28). The log says SKIP target=(12, 28) (every stand unreachable: [((11, 28), 'right'), ((13, 28), 'left')]) four times, and once earlier as not ready or no pickup. build_harvest_steps in harvest/tasks/harvest_task.py only offered those two stands. Give that tile a reachable north or south stand when the lateral stands are sealed by the plot. Add a unit test in tests/test_harvest_task.py that the step list for a blocked left and right neighbor still contains an open vertical stand. unreachable=0 with skipped=1 is this skip, not a missing target.\n";
ripe_body += "2. Barn exit. Spring D24 and Summer D9 died in the barn: directional transition timeout pos=(188,299) and pos=(188,301), tile (11, 18), tilemap 0x27. Summer D9 then never watered. The exit lives in harvest/planner/tasks/transitions.py _barn_exit_action. From tile (11, 18) the farmer must reach the barn door and change tilemap. Extend the existing barn-exit tests in tests/test_day_plan_coop.py. Do not add a new route file.\n";
ripe_body += "3. Water stand. NAV_CROP failed as water stand dist=33 limit=23 stasis=90, and the day plan deferred it until tomorrow. Summer D10 then watered 5 and skipped 3. Keep the 23px cap. The same day must repath onto a stand within 23px, or the stand target must already be that close. Do not restore soft-arrive at 33px. tests/test_crop_skills.py test_nav_crop_rejects_33px_water_stand must stay red for a 33px stand.\n";
ripe_body += "Run QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes/harvest/tests/test_harvest_task.py snes/harvest/tests/test_crop_skills.py snes/harvest/tests/test_day_plan_coop.py -q --tb=line\n";
ripe_body += "passed is true only if you read the three edits, the new harvest-stand test fails a sealed (12, 28) before the fix and passes after, the 33px water test still rejects dist 33, and pytest shows no failures. Quote the stand you added and the barn-exit change.\n";
let ripe = agent(ripe_body, #{ label: "ripe", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(ripe) {
    halt = "The ripe-crop edits did not land.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover. One emulator.\n";
    body += "Ripe evidence: ";
    body += text_of(ripe.output.evidence);
    body += "\nStart from --state LN_herd_y2_spring only. Ordinary planner, which means omit --day-plan so the label is auto_multi_day. One run. Cap --until-year 2 --until-season 1 --until-day 30 --max-frames 900000. The CLI year flag is 1-based: year 2 is ROM year byte 1. Summer D30 is the cap, not Summer D24. Do not stop the run early while the wallet is under 5300 and the date is still summer.\n";
    body += "Command shape from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 --state LN_herd_y2_spring --until-year 2 --until-season 1 --until-day 30 --max-frames 900000 --out snes/harvest/recordings/herd_cow2.json --save-end-state LN_herd_after_cow2. Tee the log to snes/harvest/logs/herd_cow2/cow2_ripe.log. Leave cow2_summer.log in place.\n";
    body += "If the JSON does not show a second cow slot, delete custom_integrations/HarvestMoon-Snes/LN_herd_after_cow2.state. Do not keep a failed pin under that name.\n";
    body += "Corn has to be shipped, including tile (12, 28) when it is mature. A potato-only wallet under 5300 is a failure. Grass must stay at 32 or more. Chickens stay 0. After the 5000 drop, the farm-gate hold names the cow. Do not press A on the shop menu while 0x096E is 0x2E. The receipt needs animal-shop tilemap 0x24, wallet down by 5000, num_cows plus 1, and a new cow slot. Count alone is not a buy.\n";
    body += "passed is true only if cows go from 1 to 2, chickens stay 0, a second slot exists, the wallet drops by 5000, planted grass is at least 32, ram_writes are 0, and mid_run_state_loads are 0. If the run ends still under 5300, do not run the summer again. Quote the end date, wallet, corn shipments, whether (12, 28) was skipped, grass, and seeds. A crash before the first overnight may be retried once from the same pin. A short wallet is not a crash.\n";
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
    body += "cow1 needs slot 0 and ram_writes 0. spring needs year 1, season 0, and a potato tile other than 0xEB. cow2 needs cows at least 2, chickens 0, a second slot, grass at least 32, and a 5000 wallet drop. accepted is true only when all three show those numbers and mid_run_state_loads 0. Quote the numbers. Do not accept a cow2 file whose end wallet is under 5300 and whose cow count is still 1.\n";
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
    if ripe != () && ripe.output != () {
        report += " Ripe: ";
        report += text_of(ripe.output.blocked);
        report += " ";
        report += text_of(ripe.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
}
let path = write_scratch_file("herd_ripe_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
