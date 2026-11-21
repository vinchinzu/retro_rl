let meta = #{
    name: "harvest-herd-exit",
    description: "Hold the farm-gate exit until the name bit is consumed",
    when_to_use: "After harvest-herd-door. Opcode 3D never targets maps 0-3. The gate counter does.",
    phases: [
        #{ title: "Exit", detail: "log the gate counter and name the cow" },
        #{ title: "Raise", detail: "one chore morning and one house sleep" },
        #{ title: "Cow2", detail: "second cow before any chicken" },
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
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "No RAM writes. No --power-on. Do not start from herd_cow1_f627470 or LN_herd_shop_receipt. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "Soft cap about 1000 lines. chicken_sale.py is at 999: do not add lines. multi_day_planner.py and day_phase_registry.py are at the cap. multi_nav.py is about 685 and may take a small edit. No new sibling module. A probe script belongs in /tmp.\n";
    p += "Controller input only. A shop menu or a return to the origin map is not a buy. Count plus a wallet drop is not a cow. SetCowNameBought at $80815F creates the slot.\n";
    return p;
}

fn exit_facts() {
    let p = "";
    p += "Start only from --state LN_herd_name_ready. Fall D21, hour 7. Money 380. num_cows 1. Slots empty. 0x7F1F5A is 0x0022, so name bit 0x0002 is set. 0x096E is 0x2E. 0x00D2 is 0x4081. 0x0196 is 0x84E1, which does not include bit 0x4000.\n";
    p += "harvest-herd-map walked town_shop_to_path+path_to_farm and stopped at the first raw tilemap 0x02, position (244, 134), 1795 frames, 0x096E still 0x2E, 0x7F1F5A 0x8022. That stop was too early. Do not stop on the first raw 0x02.\n";
    p += "Opcode 0x3D never targets maps 0, 1, 2, or 3. Do not search the event scripts again.\n";
    p += "SUB_809A64 reads Unk_Table13. Index 9 at $80B73D is bytes 00 00 00 81 88 00 58 01. Dest 0x00 plus the season is Fall map 0x02. Spawn is (0x88, 0x158), which is (136, 344), not (244, 134).\n";
    p += "While $0878 and $087A are in $C0 through $CF, the routine increments $098A. At $098A == $0D, $809D04 ORs $0196 with $4000. GameLoop calls SetScreenTransition before SUB_809A64, so the handoff is the next frame. SetScreenTransition skips RunsFunctionbyIndex when $7F1F60 bit $0040 is set and bit $0100 is clear.\n";
    p += "Indices 0-3 of UNK_TableFunctionSelect are UNK_FuncInTable. At hour 7 it jumps to CODE_83CE5B, which stores 0x096E = 0x42 when bit 0x0002 is set. CODE_82AEBA can store 0x096E from the map link byte afterward. If 0x096E goes to 0x42 and back to 0x2E, quote both frames.\n";
    p += "path_to_farm's last exit direction is right. multi_nav releases that direction when the tilemap changes. Hold right through the extra frames in the probe.\n";
    p += "Do not press A while 0x096E is 0x2E.\n";
    return p;
}

phase("Exit");
let exit_body = rules();
exit_body += "You are the gate-counter prober.\n";
exit_body += exit_facts();
exit_body += "Boot with make_harvest_env(state='LN_herd_name_ready') inside AuditedEnv, the same construction as run_to_day2. Write the probe in /tmp/herd_exit_probe.py. One process.\n";
exit_body += "Travel with the existing town_shop_to_path and path_to_farm waypoints. Do not record a new walk. When raw $22 becomes 0x02, keep holding right for 90 frames. Log a row whenever $22, $098A, $0196, $098B, $096E, $7F1F5A, or $7F1F60 changes, and include $0878, $087A, and the position on that row. Write the log to /tmp/herd_exit_log.txt.\n";
exit_body += "If $096E becomes 0x42 on any frame, stop walking and call name_entry_press until a new existing_cow_slots bit appears and the tilemap leaves 0x5F. That run is the receipt.\n";
exit_body += "If $098A reaches $0D and 60 frames later $096E is still 0x2E, stop. Quote whether bit 0x4000 was set and the value of $7F1F60. Two tries of this same hold. Do not invent a second door.\n";
exit_body += "Cap 8000 frames. Overwrite snes/harvest/recordings/herd_cow1.json. You may save LN_herd_after_cow1 only when a new slot exists. If the hold is what lets 0x096E become 0x42, add that hold to multi_nav exit_settle only while 0x7F1F5A bit 0x0002 is set and 0x096E is not 0x42. Do not change other exits.\n";
exit_body += "passed is true only if frames are greater than 0 and at most 8000, 0x096E reached 0x42, a new slot exists, num_cows stays 1, ram_writes are 0, and mid_run_state_loads are 0.\n";
let cow1 = agent(exit_body, #{ label: "exit", capability_mode: "all", output_schema: gate_schema });

let halt = "";
if !gate_ok(cow1) {
    halt = "The gate counter did not name the cow.";
}

phase("Raise");
let raised = ();
if halt == "" {
    let body = rules();
    body += "You are the cow raiser.\n";
    body += "Name evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nStart from LN_herd_after_cow1. Ordinary planner. One morning of that cow's chores, feed that slot when it is feedable, and one house-bed sleep. A calf may skip the feed penalty. Milk is not required on day 0. Happiness of that slot must not be lower after the sleep.\n";
    body += "Do not buy a chicken. Do not buy the second cow in this phase. Cap --max-frames 80000. Two tries. Overwrite snes/harvest/recordings/herd_cow1.json so the sleep numbers are in that file. passed is true only if the slot still exists, happiness did not fall, ram_writes are 0, and mid_run_state_loads are 0.\n";
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
    body += "\nThe wallet after the first cow was about 380G. Earn the second cow with the ordinary planner before any chicken. While num_cows is under 2, do not buy a chicken if the next cow can be paid for or is only waiting on grass. Need 5300G reserved and 16 grass tiles per resulting animal. One animal per day. Summer and fall only. The slot comes from the name script, not the payment.\n";
    body += "Start from the pin the raise saved. Cap --max-frames 300000. If the season ends under 5300G, set passed false and quote the wallet. Write snes/harvest/recordings/herd_cow2.json. Save LN_herd_after_cow2 only when the second slot exists.\n";
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
    body += "\nPlant Year 2 spring with the ordinary planner. ROM year byte 1, season 0. Potato or turnip only. Tile 0xEB does not count. New Year Day 1 is FestivalEscapeTask. Year 2 Spring D2 is an ordinary day. Berries stay off when the year byte is not 0. Do not flip include_field_clear.\n";
    body += "Use a clean predecessor that was not date-poked. If none exists, set passed false. Cap --until-day 6 --max-frames 400000. Write snes/harvest/recordings/y2_spring_plant.json.\n";
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
    if cow1 != () && cow1.output != () {
        report += " Exit: ";
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
let path = write_scratch_file("herd_exit_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
