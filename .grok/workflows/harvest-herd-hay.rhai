let meta = #{
    name: "harvest-herd-hay",
    description: "Refill hay before the cow starves, then buy the second cow",
    when_to_use: "After harvest-herd-ripe. Summer D30 ended at 4410G and the cow was gone.",
    phases: [
        #{ title: "Hay", detail: "pasture lip and the sealed corn tile" },
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
    p += "Repo root: /home/v/01_projects/11_games/retro. Package harvest is snes/harvest. Run tests and the game from the repo root through uv.\n";
    p += "uv run pytest fails with Failed to spawn pytest. Use QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run python -m pytest instead.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "No RAM writes. No --power-on. Do not start from LN_herd_name_ready, LN_herd_shop_receipt, LN_herd_after_raise, or herd_cow1_f627470. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. harvest_task.py is 972. multi_nav.py is 715. grass_planter.py is 855. navigation.py is 709. day_phase_registry.py, multi_day_planner.py, and chicken_sale.py are at the cap. Do not add lines to those three. No new sibling module. Do not record a new walk.\n";
    p += "Controller input only. Chickens stay 0 while cows are under 2. Leave the first-house guard in home.py, the farm-gate name hold in multi_nav.py, and WATER_STAND_ARRIVE_PX at 23. Do not turn include_field_clear on.\n";
    return p;
}

phase("Hay");
let hay_body = rules();
hay_body += "You are the hay editor. No emulator in this phase.\n";
hay_body += "Cow 1 is already in recordings/herd_cow1.json. Year 2 spring is already in recordings/y2_spring_plant.json. Do not redo either.\n";
hay_body += "Read recordings/herd_cow2.json and logs/herd_cow2/cow2_ripe.log. The run started at LN_herd_y2_spring, hay 41, planted_grass 33, one cow. It ended Y2 Summer D30 06:00 at 4410G with num_cows 0, cow_slots 0, planted_grass 26, grass_seeds 1, stored_grass 0. Corn shipments were 17. Tile (12, 28) never shipped.\n";
hay_body += "Hay fell by one bale a day. CUT_FODDER ran on Summer D1 and died: multi_nav pixel_stuck pos=(249,526) replans=4, on the way to waypoint (264, 520) after (312, 520). The grass route is the waypoint list in _build_grass in harvest/planner/day_phase_registry.py: (136, 424), (232, 392), (296, 392), (312, 488), (312, 520), (264, 520), (264, 536). Pasture bounds are (13, 33)-(18, 38). (249, 526) is the row north of that pasture. PLANT_PASTURE hit the same pixel on every later try. The cow was fed through Summer D18. On D19 hay was 0, fed=0, and talk still paid +1. Happiness then fell about 7 a night (62, 43, 36, 29, 22, 15, 8) and the slot was empty after D25. Unfed is -8. Do not poke happiness. Refill the bin from mature pasture before the bale count hits 0, and leave planted_grass at 32 or more.\n";
hay_body += "Summer D5 bought one grass bag, 3170 to 2670, because planted_grass had already fallen under 32. The seed was never planted. logs/herd_cow2/cow2_summer.log completed CUT_FODDER twice from the same pin family and ended at planted_grass 33 and stored_grass 59. Find why this approach sticks at (249, 526) and fix that stall. An in-place coordinate change is allowed. Do not append waypoints in day_phase_registry.py.\n";
hay_body += "Harvest tile (12, 28) is still skipped. cow2_ripe.log says every stand unreachable: ((12, 29), up) and ((12, 27), down). Those are the vertical stands the last edit added, and they are sealed by the plot. Do not add another adjacent notch. When the only stands are other ripe tiles, pick the ring first and retry the center after a neighbor is gone. A permanent skip of (12, 28) is a failure. harvest_task.py must stay at or under 1000 lines.\n";
hay_body += "Add a unit test that a center tile boxed by ripe neighbors is harvested once one neighbor is cleared, and a unit test that the pasture approach does not treat pixel (249, 526) as arrival on the way to (264, 520). Run uv run python -m pytest on snes/harvest/tests/test_harvest_task.py and the grass or day-plan test you touch, with QT_QPA_PLATFORM=offscreen and RETRO_RL_TEST_TIER=game-no-rom.\n";
hay_body += "passed is true only if both tests pass, WATER_STAND_ARRIVE_PX is still 23, and you quote the stall fix and the retry rule.\n";
let hay = agent(hay_body, #{ label: "hay", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(hay) {
    halt = "The hay edits did not land.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover. One emulator.\n";
    body += "Hay evidence: ";
    body += text_of(hay.output.evidence);
    body += "\nStart from --state LN_herd_y2_spring only. Omit --day-plan so the label is auto_multi_day. One run. Cap --until-year 2 --until-season 1 --until-day 30 --max-frames 900000. The CLI year flag is 1-based.\n";
    body += "Command from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 --state LN_herd_y2_spring --until-year 2 --until-season 1 --until-day 30 --max-frames 900000 --out snes/harvest/recordings/herd_cow2.json --save-end-state LN_herd_after_cow2. Tee the log to snes/harvest/logs/herd_cow2/cow2_hay.log. Leave cow2_ripe.log and cow2_summer.log in place.\n";
    body += "Watch the log while it runs. If Summer D1 CUT_FODDER or PLANT_PASTURE dies again at pixel (249, 526), or hay reaches 0, stop that process. That is a failed fix. Do not keep the summer going and do not start a second one.\n";
    body += "If the JSON does not show a second cow slot, delete custom_integrations/HarvestMoon-Snes/LN_herd_after_cow2.state.\n";
    body += "The receipt needs cows 1 to 2, chickens 0, a new cow slot, animal-shop tilemap 0x24, a 5000 wallet drop, planted grass at least 32, the first cow still present, ram_writes 0, and mid_run_state_loads 0. Count alone is not a buy. Do not press A on the shop menu while 0x096E is 0x2E. A short wallet is not a retry. Quote the end date, wallet, hay, planted grass, corn shipments, whether (12, 28) shipped, and the cow slot count.\n";
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
    body += "cow1 needs slot 0 and ram_writes 0. spring needs year 1, season 0, and a potato tile other than 0xEB. cow2 needs cows at least 2, chickens 0, two slots, grass at least 32, stored hay above 0, and a 5000 wallet drop. accepted is true only when all three show those numbers and mid_run_state_loads 0. A cow2 file whose cow count is 0, or whose end wallet is under 5300 with one cow, is rejected. Quote the numbers.\n";
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
    if hay != () && hay.output != () {
        report += " Hay: ";
        report += text_of(hay.output.blocked);
        report += " ";
        report += text_of(hay.output.evidence);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
        report += " ";
        report += text_of(cow2.output.evidence);
    }
}
let path = write_scratch_file("herd_hay_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
