let meta = #{
    name: "harvest-herd-pasture",
    description: "Plant the saved Summer D25 grass, then buy and raise two cows and open Year 2 spring",
    when_to_use: "After harvest-herd-spring. The D25 pin has 4 grass seeds and 0 planted tiles. Do not power on again.",
    phases: [
        #{ title: "Plant", detail: "make the 4 seeds raise planted_grass" },
        #{ title: "Cow1", detail: "32 tiles, 5300G, buy and raise" },
        #{ title: "Cow2", detail: "second cow" },
        #{ title: "Spring", detail: "Year 2 spring planting" },
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
    "required": ["accepted", "plant", "cow1", "cow2", "spring", "evidence"],
    "properties": #{
        "accepted": #{ "type": "boolean" },
        "plant": #{ "type": "boolean" },
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
    p += "Read snes/harvest/AGENTS.md and snes/harvest/harvest/tasks/grass_planter.py and snes/harvest/harvest/planner/day_phase_registry.py _build_grass before you edit.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "Clean: controller input only. RAM writes forbidden. A mid-run state load does not count. Do not start Day 2 from Y1_D2_Morning_After_D1. No --power-on. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4.\n";
    p += "Command shape, from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 <flags>.\n";
    p += "The saved morning is --state herd_cow1_f627470. It is Year 1 Summer D25 06:00, tilemap 21, money 3010, num_cows 0, planted_grass 0, grass_seeds 4. The file is snes/harvest/custom_integrations/HarvestMoon-Snes/herd_cow1_f627470.state. A copy is snes/harvest/logs/herd_cow1/latest.state.\n";
    p += "Do not boot from power-on. The last power-on already spent 627470 frames and ended on this pin.\n";
    p += "Measured miss, from snes/harvest/recordings/herd_cow1.json: Summer D2-D20 plant pixel_stuck at (313,537). Fall D2 PLANT_PASTURE tilled 15 and planted 0. Later days said pasture progress 0->0 while the 4 seeds fell to 0. The run then wedged on tilemap 0x09 text 0x024F. Do not repeat that town dialogue.\n";
    p += "PASTURE_BOUNDS in grass_planter.py is tiles (13, 33) through (18, 38). Pixel (312, 536) is the last waypoint in _build_grass and sits on tile (19, 33), one tile east of that rectangle. That waypoint is the stall.\n";
    p += "Grass seeds plant a 3x3 while the farmer stands on fresh tilled tile 0x07. planted_grass is the ROM counter at the planted_grass field. A seed-count drop with planted_grass unchanged is a failure, not a plant.\n";
    p += "day_phase_registry.py is 999 lines. Change the existing waypoint tuple in place. Do not add lines to that file. grass_planter.py is about 750 lines and can take a small fix. No new sibling module.\n";
    p += "Animal buys stay in summer and fall. Cow price is 5000G. The planner reserve is 5300G. One animal per day. A buy needs the shop tilemap, the wallet down, num_cows up by 1, and a new initialized slot.\n";
    p += "Two honest tries on a ROM receipt, then stop. Overwrite the one JSON path named in your task. Do not mint a _vN report.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest <path> -q --tb=line.\n";
    return p;
}

fn gate_prompt(role, body) {
    let p = rules();
    p += "You are ";
    p += role;
    p += ".\n";
    p += body;
    p += "\nReturn passed, evidence, frames, report_path, and blocked. passed is true only with a file path you opened and numbers you read from it.\n";
    return p;
}

phase("Plant");
let plant = agent(
    gate_prompt(
        "the pasture planter",
        "Fix the ordinary PLANT_PASTURE path so the 4 seeds on herd_cow1_f627470 raise planted_grass.\n"
            + "Read the stall before you edit. The farmer must walk inside PASTURE_BOUNDS, hoe 0x07, stand on that tile, and press the seed item so a tile becomes 0x70 and planted_grass rises. Leaving the last waypoint at (312, 536) repeats the stall.\n"
            + "Prove it with one morning from that pin. Cap --max-frames 60000. Write snes/harvest/recordings/herd_pasture.json. You may --save-end-state LN_herd_pasture only when planted_grass is greater than 0.\n"
            + "Stay on the farm. If tilemap 0x09 or text 0x024F appears, that try failed.\n"
            + "passed is true only if you read frames greater than 0, planted_grass greater than the start value, grass_seeds lower only together with that rise, at least one grass tile 0x70 inside the pasture, ram_writes 0, and mid_run_state_loads 0.\n"
            + "A no-ROM test may pin the waypoint inside the pasture bounds. Run it. Two tries, then passed false."
    ),
    #{ label: "plant", capability_mode: "all", output_schema: gate_schema }
);

let halt = "";
if !gate_ok(plant) {
    halt = "Pasture plant failed.";
}

phase("Cow1");
let cow1 = ();
if halt == "" {
    let body = "";
    body += "Plant evidence: ";
    body += text_of(plant.output.evidence);
    body += "\nFrom LN_herd_pasture, or from herd_cow1_f627470 if that save was not written, let the ordinary planner finish the pasture and buy the first cow.\n";
    body += "Need planted_grass at least 32 and money at least 5300 before BUY_COW. This pin started at 3010G. Fall shipping may close the gap. Do not RAM-poke gold. Do not buy grass seeds unless the wallet stays at or above 5300 after the buy. Do not buy a chicken.\n";
    body += "Cap --max-frames 300000. If money is still under 5300 when Fall D20 ends, set passed false and quote the wallet. Do not walk into tilemap 0x09.\n";
    body += "After the buy, run the next morning's cow chores and one house-bed sleep. Feed that slot's bit when the cow is feedable. Happiness must not fall. A calf may skip the feed penalty.\n";
    body += "Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when the receipt passes.\n";
    body += "passed is true only if you read frames greater than 0, cows 0 to 1, a new slot, wallet down by the cow price, planted_grass at least 32, shop tilemap, happiness not down, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false.\n";
    cow1 = agent(gate_prompt("the first-cow prover", body), #{ label: "cow1", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow1) {
        halt = "First cow buy and raise failed.";
    }
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = "";
    body += "Cow 1 evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nBuy the second cow before any chicken. next_livestock returns cow when num_cows is 1 and the shop gates pass, including planted grass of 16 times the animal total after the buy. While num_cows is under 2, do not return chicken. Do not schedule a third cow. Summer and fall only.\n";
    body += "Start from LN_herd_after_cow1. Cap --max-frames 180000. Write snes/harvest/recordings/herd_cow2.json. You may --save-end-state LN_herd_after_cow2 only when it passes.\n";
    body += "passed is true only if you read frames greater than 0, cows 1 to 2, a second initialized slot, wallet down by the cow price, cow 1 still present, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false. Do not break the pasture plant.\n";
    cow2 = agent(gate_prompt("the second-cow prover", body), #{ label: "cow2", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Spring");
let spring = ();
if halt == "" {
    let body = "";
    body += "Cow 2 evidence: ";
    body += text_of(cow2.output.evidence);
    body += "\nGet the ordinary planner through the start of Year 2 spring planting. Year byte 1, season 0. Potato or turnip only. Tile 0xEB does not count. Year 2 Spring D2 stays an ordinary day. Berry runs stay off when year is not 0. Do not change the pasture target or next_livestock.\n";
    body += "New Year on Year 2 Spring D1 is a forced festival. Use a clean Year 2 spring pin if one exists. LN_calendar_Y2_Spring_D1 and LN_calendar_Y2_Spring_D2 count only if they were not date-poked. If none is clean, set passed false and name the missing pin. Do not edit the calendar.\n";
    body += "Cap --max-frames 400000 --until-day 6. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "passed is true only if you read frames greater than 0, year 1, season 0, at least one potato or turnip tile other than 0xEB, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false.\n";
    spring = agent(gate_prompt("the year-2 spring planter", body), #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = "";
    body += "Read only. Open snes/harvest/recordings/herd_pasture.json, herd_cow1.json, herd_cow2.json, and y2_spring_plant.json.\n";
    body += "plant is true only if herd_pasture.json shows planted_grass up and ram_writes 0 and mid_run_state_loads 0.\n";
    body += "cow1 is true only if herd_cow1.json shows cows ending at least 1, wallet down, planted_grass at least 32, ram_writes 0, mid_run_state_loads 0.\n";
    body += "cow2 is true only if herd_cow2.json shows cows ending at least 2, wallet down, ram_writes 0, mid_run_state_loads 0.\n";
    body += "spring is true only if y2_spring_plant.json shows year 1, season 0, a potato or turnip tile other than 0xEB, ram_writes 0, mid_run_state_loads 0.\n";
    body += "accepted is true only when all four are true. Quote the numbers.\n";
    verdict = agent(rules() + "You are the verifier.\n" + body, #{ label: "verify", capability_mode: "read-only", output_schema: verdict_schema });
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
    if plant != () && plant.output != () {
        report += " Plant: ";
        report += text_of(plant.output.blocked);
        report += " ";
        report += text_of(plant.output.evidence);
    }
    if cow1 != () && cow1.output != () {
        report += " Cow1: ";
        report += text_of(cow1.output.blocked);
    }
    if cow2 != () && cow2.output != () {
        report += " Cow2: ";
        report += text_of(cow2.output.blocked);
    }
    if spring != () && spring.output != () {
        report += " Spring: ";
        report += text_of(spring.output.blocked);
    }
}
let path = write_scratch_file("herd_pasture_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
