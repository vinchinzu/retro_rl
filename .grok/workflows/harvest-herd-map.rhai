let meta = #{
    name: "harvest-herd-map",
    description: "Walk the paid cow from town onto the farm so the name script starts",
    when_to_use: "After harvest-herd-slot. The name bit is set. The shop script is not running.",
    phases: [
        #{ title: "Name", detail: "enter the farm and name the cow" },
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
    p += "Read bank_81.asm SUB_81BFB7, PlayerActionPointerTable, and CODE_81DE8A, and bank_83.asm RunsFunctionbyIndex and CODE_83CE5B, before you press a button.\n";
    p += "Do not edit STATUS.md, plan.md, or the living residual. Do not run bd. Do not commit. No RAM writes. No power-on. No replay from herd_cow1_f627470.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4.\n";
    p += "Start only from --state LN_herd_name_ready. Measured: tilemap 0x06, player (600, 296), money 380, num_cows 1, slots empty, planted_grass 33, text 0x0305, inputstate 1, 0x019B 0, 0x096E 0x2E, 0x096F 1, 0x00D4 0, 0x00D2 0x4081, 0x7F1F5A 0x22.\n";
    p += "0x096E is not consulted while 0x00D2 bit 0x0040 is clear. SUB_81BFB7 sends that frame to player_action 0x00D4, which is 0. Waiting on this pin cannot open the name. Do not press A. Choice 0 on action 0x2E sees 0x7F1F5A bit 0x0020 and shows already-bought text 0x03B6.\n";
    p += "0x7F1F5A bit 0x0002 is set. CODE_83CE5B is the handoff: it stores 0x096E = 0x42, clears 0x096F, and sets 0x00D2 bit 0x0040. RunsFunctionbyIndex calls that handoff for tilemaps 0, 1, 2, and 3. Tilemap 0x06 uses a different table entry. Town maps 0x05, 0x06, and 0x07 are the town. The farm is tilemap 0x00.\n";
    p += "Walk from 0x06 onto tilemap 0x00 with the existing town-to-farm route. Do not record a new walk. When 0x096E becomes 0x42, use name_entry_press. The slot is created by SetCowNameBought. If you are on tilemap 0 and 0x096E is still 0x2E, stop and quote 0x7F1F5A, 0x096E, and 0x00D2.\n";
    p += "Cap --max-frames 8000. Two tries from the same pin. Stay under the 1000-line file cap. No new module.\n";
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
        "the farm-entry namer",
        "Leave tilemap 0x06 for the farm and let CODE_83CE5B start the name. Then confirm the name and sleep one house night without dropping the new slot's happiness.\n"
            + "Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when the slot exists.\n"
            + "passed is true only if frames are greater than 0 and at most 8000, num_cows stays 1, a new slot exists, 0x096E reached 0x42, ram_writes are 0, and mid_run_state_loads are 0.\n"
            + "Two tries, then passed false."
    ),
    #{ label: "name", capability_mode: "all", output_schema: gate_schema }
);

let halt = "";
if !gate_ok(cow1) {
    halt = "Farm entry did not name the cow.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = "Cow 1 evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nThe wallet on the name pin was 380G. Earn the second cow with the ordinary planner before any chicken. Need 5300G and grass for 16 times the animal total after the buy. The slot is created by the name script, not by the payment.\n";
    body += "Start from LN_herd_after_cow1. Cap --max-frames 300000. If the season ends under 5300G, set passed false and quote the wallet. Write snes/harvest/recordings/herd_cow2.json.\n";
    body += "passed is true only if cows go 1 to 2, a second slot exists, the wallet drops by 5000, ram_writes are 0, and mid_run_state_loads are 0.\n";
    cow2 = agent(gate_prompt("the second-cow prover", body), #{ label: "cow2", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Spring");
let spring = ();
if halt == "" {
    let body = "Cow 2 evidence: ";
    body += text_of(cow2.output.evidence);
    body += "\nPlant Year 2 spring. Year byte 1, season 0. Potato or turnip only. Tile 0xEB does not count. Use a clean Year 2 spring pin that was not date-poked. If none exists, set passed false. Cap --max-frames 400000 --until-day 6. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "passed is true only if frames are greater than 0, year is 1, season is 0, one real crop tile exists, ram_writes are 0, and mid_run_state_loads are 0.\n";
    spring = agent(gate_prompt("the year-2 spring planter", body), #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = "Read only. Open the three receipts under snes/harvest/recordings/. cow1 needs a slot and ram_writes 0. cow2 needs cows at least 2 and a wallet drop. spring needs year 1, season 0, and a potato or turnip tile other than 0xEB. accepted is true only when all three are true. Quote the numbers.\n";
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
let path = write_scratch_file("herd_map_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
