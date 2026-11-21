let meta = #{
    name: "harvest-herd-slot",
    description: "Name the cow from the armed 096F pin, then the second cow and Year 2 spring",
    when_to_use: "After harvest-herd-name. The payment is done. Start at LN_herd_name_ready, where 096F is already 1.",
    phases: [
        #{ title: "Name", detail: "let 096F create the slot" },
        #{ title: "Cow2", detail: "second cow after the wallet recovers" },
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
    p += "Read snes/harvest/AGENTS.md, snes/harvest/docs/perfect_rancher/research/F3_herd.md section Buy value, HM-Decomp bank_81.asm at CODE_81E08A and CODE_81E6EE, and tasks/buy_cow.json before you press a button.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or the living residual. Do not run bd. Do not commit.\n";
    p += "Clean: controller input only. No RAM writes. No power-on. No replay from herd_cow1_f627470. That fall already ran three times.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4.\n";
    p += "The only start pin is --state LN_herd_name_ready. It is a copy of logs/herd_name/progress_f174000.state. Measured RAM: Year 1 Fall D21 07:09, tilemap 0x06, player (600, 296), money 380, num_cows 1, cow slots empty, planted_grass 33, dialog_text_id 0x0305, input_lock 1, inputstate 0x019A = 1, 0x019B = 0, 0x096F = 1.\n";
    p += "0x096F = 1 means the bought-name machine is armed. CODE_81E732 runs on that value, opens text 0x030C or 0x010E, sets inputstate to 2, and stores 0x096F = 2. When inputstate leaves 2, CODE_81E76E stores 0x096F = 3, 0x099F = 1, and ORs 0x0196 with 0x2000. SetCowNameBought at 0x80815F calls AddNewCow during the name fade. The slot does not exist at payment. Payment already happened at CODE_81E08A.\n";
    p += "Do not start from LN_herd_shop_receipt or logs/herd_name/latest.state. Those are later: 0x096F is 0, money is 80, text is 0x0304, and the name machine is gone.\n";
    p += "A fresh day plan sees num_cows already 1 and will walk away. Do not let nav leave tilemap 0x06 before the slot exists. Scan tasks/buy_cow.json for the buttons after the wallet drop. Do not record a new tape.\n";
    p += "chicken_sale.py is about 950 lines. Edit CowPurchaseTask in place. Stay under about 1000 lines. No new module.\n";
    p += "Cap the name window at --max-frames 12000. Two tries from the same pin. If 0x096F is still 1 after 600 frames, stop that try and quote 0x096F, the text id, inputstate, and 0x0196. Do not wander.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest <path> -q --tb=line.\n";
    return p;
}

fn gate_prompt(role, body) {
    let p = rules();
    p += "You are ";
    p += role;
    p += ".\n";
    p += body;
    p += "\nReturn passed, evidence, frames, report_path, and blocked. passed is true only with numbers you read from a file you opened.\n";
    return p;
}

phase("Name");
let cow1 = agent(
    gate_prompt(
        "the cow-slot prover",
        "Finish the armed name from LN_herd_name_ready. The slot must appear the way SetCowNameBought writes it. Then leave the name screen and sleep one house night without dropping that slot's happiness. A bought cow starts as a child, so feed may not apply yet.\n"
            + "Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when the slot exists.\n"
            + "passed is true only if you read frames greater than 0 and at most 12000, num_cows stays 1, a new cow slot exists, the name screen has closed, ram_writes 0, and mid_run_state_loads 0. Happiness after the house sleep must be at least the value at slot creation.\n"
            + "Run the no-ROM purchase test you touch before the ROM. Two tries, then passed false."
    ),
    #{ label: "name", capability_mode: "all", output_schema: gate_schema }
);

let halt = "";
if !gate_ok(cow1) {
    halt = "Cow slot was not created.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = "";
    body += "Cow 1 evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nThe shop pin had 380G after the first cow. Earn a second cow only with the ordinary planner, before any chicken. Need 5300G and enough planted grass for 16 times the animal total after the buy. The same slot rule applies: payment is not the slot.\n";
    body += "Start from LN_herd_after_cow1. Cap --max-frames 300000. If Fall ends under 5300G, set passed false and quote the wallet. Write snes/harvest/recordings/herd_cow2.json.\n";
    body += "passed is true only if cows go 1 to 2, a second slot exists, the wallet drops by 5000, ram_writes 0, and mid_run_state_loads 0.\n";
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
    body += "\nPlant Year 2 spring with the ordinary planner. Year byte 1, season 0. Potato or turnip only. Tile 0xEB does not count. Do not change the cow name path. Use a clean Year 2 spring pin that was not date-poked. If none exists, set passed false. Cap --max-frames 400000 --until-day 6. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "passed is true only if frames are greater than 0, year is 1, season is 0, one real potato or turnip tile exists, ram_writes are 0, and mid_run_state_loads are 0.\n";
    spring = agent(gate_prompt("the year-2 spring planter", body), #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = "Read only. Open snes/harvest/recordings/herd_cow1.json, herd_cow2.json, and y2_spring_plant.json. ";
    body += "cow1 is true only if a cow slot exists, cows are at least 1, ram_writes are 0, and mid_run_state_loads are 0. ";
    body += "cow2 is true only if cows are at least 2, the wallet dropped, ram_writes are 0, and mid_run_state_loads are 0. ";
    body += "spring is true only if year is 1, season is 0, and a potato or turnip tile is not 0xEB. ";
    body += "accepted is true only when all three are true. Quote the numbers.\n";
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
    if cow1 != () && cow1.output != () {
        report += " Name: ";
        report += text_of(cow1.output.blocked);
        report += " ";
        report += text_of(cow1.output.evidence);
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
let path = write_scratch_file("herd_slot_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
