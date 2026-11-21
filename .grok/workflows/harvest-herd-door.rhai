let meta = #{
    name: "harvest-herd-door",
    description: "Cross a real farm transition so the paid cow name script starts",
    when_to_use: "After harvest-herd-map. The path walk reached tilemap 2 and left 096E at 2E.",
    phases: [
        #{ title: "Door", detail: "find the transit opcode that arms the name" },
        #{ title: "Name", detail: "cross that door and name the cow" },
        #{ title: "Raise", detail: "one chore morning and one house sleep" },
        #{ title: "Cow2", detail: "second cow before any chicken" },
        #{ title: "Spring", detail: "Year 2 spring planting" },
        #{ title: "Verify", detail: "fail closed on the JSON" },
    ],
};

let door_schema = #{
    "type": "object",
    "required": ["found", "evidence"],
    "properties": #{
        "found": #{ "type": "boolean" },
        "dest": #{ "type": "integer" },
        "how_to_cross": #{ "type": "string" },
        "evidence": #{ "type": "string" },
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

fn door_ok(r) {
    if r == () || r.success != true || r.output == () { return false; }
    if r.output.found != true { return false; }
    if type_of(r.output.evidence) != "string" { return false; }
    if r.output.evidence == "" { return false; }
    return true;
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
    p += "Repo root: /home/v/01_projects/11_games/retro. Package harvest is snes/harvest. Run tools from the repo root with uv run.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "No RAM writes. No --power-on. Do not start from herd_cow1_f627470 or LN_herd_shop_receipt. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. multi_day_planner.py is at the cap: do not add lines. day_phase_registry.py is at the cap: edit existing tuples in place. No new sibling module.\n";
    p += "A shop menu or a cross-map return to the origin is not a buy. Count plus a 5000G wallet drop is not a cow. The slot is created by SetCowNameBought at $80815F during the name fade.\n";
    return p;
}

fn rom_facts() {
    let p = "";
    p += "Start pin: --state LN_herd_name_ready under custom_integrations/HarvestMoon-Snes. Year 1 Fall D21 about 07:09. Tilemap raw 6 canon 4, pos about (600, 368). Money 380. num_cows 1. Cow slots empty. planted_grass 33. Text 0x0305. 0x096E is 0x2E. 0x096F is 1. 0x00D4 is 0. 0x00D2 is 0x4081. 0x0196 is 0x84E1. 0x7F1F5A is 0x0022. Bit 0x0002 of 0x7F1F5A is the pending name. Bit 0x0020 means already bought today.\n";
    p += "harvest-herd-map already walked town_shop_to_path+path_to_farm. Both tries were 1795 frames and ended on raw tilemap 0x02 at (244, 134). 0x096E stayed 0x2E. 0x7F1F5A became 0x8022. 0x00D2 stayed 0x4081. Slots stayed empty. That arrival is not the name. Do not stop on it again.\n";
    p += "SetScreenTransition is $809671 in bank_80.asm. If 0x0196 bit 0x4000 is clear it JMPs to ScreenTransitionReturn and never calls RunsFunctionbyIndex. The pin's 0x84E1 does not have bit 0x4000. ScreenFadein at $8087CE ORs 0x8000 onto 0x7F1F5A. That is why the walk showed 0x8022. The name bit 0x0002 was still set.\n";
    p += "RunsFunctionbyIndex at $83CA98 indexes UNK_TableFunctionSelect by tilemap_to_load. Indices 0, 1, 2, and 3 are $CB69, which is UNK_FuncInTable. At 07:09 the hour check at $83CC53 JMPs to CODE_83CE5B. That block, when 0x7F1F5A bit 0x0002 is set, stores 0x096E = 0x42, clears 0x096F, and ORs 0x00D2 with 0x0040.\n";
    p += "Opcode 0x3D is Event_TransitiontoX at $84B52D. It stores the next script byte into transition_dest 0x098B and ORs 0x0196 with 0x4000. Opcode 0x3C is Event_TransitiontoHouse at $84B503. It stores dest 0x15 and also sets the bit. Dest 0x15 is not index 0-3, so entering the house does not run the name function. harvest/scripts/event_dis.py names opcode 0x3D TRANSIT and gives it one operand byte.\n";
    p += "Do not press A while 0x096E is 0x2E. Choice 0 shows already-bought text 0x03B6. B at $81E222 closes text and does not switch the action to 0x42. 0x096E is ignored while 0x00D2 bit 0x0040 is clear.\n";
    return p;
}

phase("Door");
let door_prompt = rules();
door_prompt += "You are the door finder. Read only. Do not boot the emulator.\n";
door_prompt += rom_facts();
door_prompt += "Use grep and read_file on bank_80.asm, bank_83.asm, bank_84.asm, and harvest/scripts/event_dis.py. Disassemble with: uv run python -m harvest.scripts.event_dis\n";
door_prompt += "Find one event script that executes opcode 0x3D with operand 0x00, 0x01, 0x02, or 0x03, and name an existing route or landmark that stands on that trigger. House door landmark is map_config house_door tile (8, 26) on the farm. farm_to_house is an existing route.\n";
door_prompt += "Quote the script address and the operand byte. found is true only with that quote. how_to_cross names the existing route and the door, including whether the player must leave the house after opcode 0x3C. If you cannot find the script, found is false.\n";
let door = agent(door_prompt, #{ label: "door", capability_mode: "read-only", output_schema: door_schema });

let halt = "";
if !door_ok(door) {
    halt = "No farm transit opcode was found.";
}

phase("Name");
let cow1 = ();
if halt == "" {
    let body = rules();
    body += "You are the door crosser.\n";
    body += rom_facts();
    body += "Door finder evidence: ";
    body += text_of(door.output.evidence);
    body += "\nHow to cross: ";
    body += text_of(door.output.how_to_cross);
    body += "\nDest byte: ";
    body += text_of(door.output.dest);
    body += "\n";
    body += "From LN_herd_name_ready only. Travel with existing routes. Do not record a new walk. The measurement is a frame where 0x0196 bit 0x4000 is set and 0x098B is 0, 1, 2, or 3. The following frames must show 0x096E = 0x42. Log 0x22, 0x098B, 0x0196, 0x096E, 0x00D2, and 0x7F1F5A on each tilemap change.\n";
    body += "The path arrival at raw tilemap 2 with 0x096E still 0x2E is not that frame. Keep going to the transit the finder named. Entering the house with opcode 0x3C is not the name. Leave through the 0x3D back onto the farm.\n";
    body += "When 0x096E becomes 0x42, use name_entry_press from harvest.core.scene. Then confirm the slot with existing_cow_slots. Cap --max-frames 8000. Two tries of that same door. If the bit gets set, dest is 0-3, and 0x096E stays 0x2E, stop and quote the six bytes.\n";
    body += "Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when a new slot exists. passed is true only if frames are greater than 0 and at most 8000, 0x096E reached 0x42, a new slot exists, num_cows stays 1, ram_writes are 0, and mid_run_state_loads are 0.\n";
    cow1 = agent(body, #{ label: "name", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow1) {
        halt = "The farm transit did not name the cow.";
    }
}

phase("Raise");
let raised = ();
if halt == "" {
    let body = rules();
    body += "You are the cow raiser.\n";
    body += "Name evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nStart from LN_herd_after_cow1. Ordinary planner. One morning of that cow's chores, feed that slot when it is feedable, and one house-bed sleep. A calf may skip the feed penalty. Milk is not required on day 0. Happiness of that slot must not be lower after the sleep.\n";
    body += "Do not buy a chicken. Do not buy the second cow in this phase. Cap --max-frames 80000. Two tries. Overwrite snes/harvest/recordings/herd_cow1.json with the raise included, or write the sleep numbers into that same file if the name block is already there. passed is true only if the slot still exists, happiness did not fall, ram_writes are 0, and mid_run_state_loads are 0.\n";
    raised = agent(body, #{ label: "raise", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(raised) {
        halt = "The named cow was not raised.";
    }
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = rules();
    body += "You are the second-cow prover.\n";
    body += "Raise evidence: ";
    body += text_of(raised.output.evidence);
    body += "\nThe wallet after the first cow was about 380G. Earn the second cow with the ordinary planner before any chicken. While num_cows is under 2, do not buy a chicken if the next cow can be paid for or is only waiting on grass. Need 5300G reserved and 16 grass tiles per resulting animal. One animal per day. Summer and fall only. Rain blocks delivery. Do not schedule a third cow.\n";
    body += "The slot is created by the name script, not by the payment. AnimalPurchaseSession success on count and wallet is not the receipt.\n";
    body += "Start from the pin the raise saved. Cap --max-frames 300000. If the season ends under 5300G, set passed false and quote the wallet. Write snes/harvest/recordings/herd_cow2.json. You may --save-end-state LN_herd_after_cow2 only when the second slot exists.\n";
    body += "passed is true only if cows go from 1 to 2, a second new slot exists, the wallet drops by 5000, tilemap was 0x24 at the payment, ram_writes are 0, and mid_run_state_loads are 0.\n";
    cow2 = agent(body, #{ label: "cow2", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(cow2) {
        halt = "Second cow receipt failed.";
    }
}

phase("Spring");
let spring = ();
if halt == "" {
    let body = rules();
    body += "You are the year-2 spring planter.\n";
    body += "Cow 2 evidence: ";
    body += text_of(cow2.output.evidence);
    body += "\nPlant Year 2 spring with the ordinary planner. ROM year byte 1, season 0. Potato or turnip only. Tile 0xEB is a failed off-season plant and does not count. New Year Day 1 is FestivalEscapeTask. Year 2 Spring D2 is an ordinary day. Berries stay off when the year byte is not 0. Power-on still forces include_field_clear off. Do not flip that.\n";
    body += "Use a clean predecessor that was not date-poked. If none exists, set passed false and say so. Cap --until-day 6 --max-frames 400000. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "passed is true only if frames are greater than 0, year is 1, season is 0, at least one potato or turnip tile exists and is not 0xEB, ram_writes are 0, and mid_run_state_loads are 0.\n";
    spring = agent(body, #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = rules();
    body += "You are the verifier. Read only. Open snes/harvest/recordings/herd_cow1.json, herd_cow2.json, and y2_spring_plant.json.\n";
    body += "cow1 needs a new slot, a house sleep whose happiness did not fall, and ram_writes 0. cow2 needs cows at least 2, a second slot, and a 5000 wallet drop. spring needs year 1, season 0, and a potato or turnip tile other than 0xEB. accepted is true only when all three files show those numbers and mid_run_state_loads 0. Quote the numbers.\n";
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
    if door != () && door.output != () {
        report += " Door: ";
        report += text_of(door.output.evidence);
    }
    if cow1 != () && cow1.output != () {
        report += " Name: ";
        report += text_of(cow1.output.blocked);
        report += " ";
        report += text_of(cow1.output.evidence);
    }
    if raised != () && raised.output != () {
        report += " Raise: ";
        report += text_of(raised.output.blocked);
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
let path = write_scratch_file("herd_door_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
