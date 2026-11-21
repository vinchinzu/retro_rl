let meta = #{
    name: "harvest-herd-name",
    description: "Finish the cow name so the new slot exists, then the second cow and Year 2 spring",
    when_to_use: "After harvest-herd-pasture. The Fall D21 buy raised num_cows and left cow_slots at 0 inside the shop.",
    phases: [
        #{ title: "Name", detail: "name the cow and leave the shop" },
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
    p += "Read snes/harvest/AGENTS.md, harvest/planner/tasks/chicken_sale.py CowPurchaseTask, harvest/planner/tasks/sale_menu.py AnimalPurchaseSession, and harvest/core/scene.py name_entry_press before you edit.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md, snes/harvest/docs/plan.md, or snes/harvest/docs/tasks/rr-20w-run16-defects.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "Clean: controller input only. RAM writes forbidden. A mid-run state load does not count inside the receipt run. Do not power on. No --perfect-rancher. No --day-plan buy_cow.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4.\n";
    p += "Command shape, from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 <flags>.\n";
    p += "Predecessor is --state herd_cow1_f627470. Year 1 Summer D25 06:00, money 3010, cows 0, grass seeds 4, planted_grass 0. There is no saved pin of the shop buy. Do not invent one.\n";
    p += "Pasture already plants. snes/harvest/recordings/herd_pasture.json shows frames 29134, planted_grass 0 to 2, grass_seeds 4 to 2, ram_writes 0. Do not redo that waypoint work.\n";
    p += "The failed buy is snes/harvest/recordings/herd_cow1.json: frames 232901, end Y1 Fall D21, tilemap 0x24, money 80, num_cows 1, cow_slots 0, planted_grass 33, grass_seeds 0, ram_writes 0, mid_run_state_loads 0. Scene pos (23,186). Exit recovery could not dismiss text 0x0341 at (47,182), and unknown-map timeouts at (177,180) and (23,186). animal_shop_to_town leaves through (137,212) facing down. Bare A did not clear 0x0341. Do not repeat bare A.\n";
    p += "CowPurchaseTask returns success only when a new cow slot exists. While the count has risen and the slot is missing and the tilemap is not 0x5F, the current step waits with an empty action. That idle hits the timeout, the optional buy phase ends, and EXIT_ANIMAL_SHOP inherits text 0x0341. Name entry is name_entry_press and runs only on tilemap 0x5F.\n";
    p += "chicken_sale.py is about 876 lines. Put the post-buy advance in CowPurchaseTask.step. Do not add a sibling module. Stay under about 1000 lines.\n";
    p += "A cow receipt needs the shop tilemap, the wallet down by 5000, num_cows up by 1, and a new initialized slot after the name screen closes. Then leave 0x24. The next morning runs cow chores and one house-bed sleep. Happiness on that slot must not fall. A calf may skip the feed penalty.\n";
    p += "Two ROM tries. The first starts at herd_cow1_f627470 with --checkpoint-dir snes/harvest/logs/herd_name --checkpoint-on-progress --max-frames 280000. The second try may load only the latest checkpoint that is already inside tilemap 0x24 or on the buy morning. That load is the start of the second try, and that try's own mid_run_state_loads must be 0. Do not replay the fall if that checkpoint exists.\n";
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

phase("Name");
let cow1 = agent(
    gate_prompt(
        "the cow-name prover",
        "Make CowPurchaseTask finish the name before it hands the day to the shop exit.\n"
            + "Read the animal-shop script and the bytes that keep text 0x0341 on screen. Advance that text until tilemap 0x5F, then use name_entry_press. Succeed only when existing_cow_slots shows a new slot, the tilemap is not 0x5F, and input_lock is 1 so the farmer can walk. The exit route then uses animal_shop_to_town. Do not succeed on the count change alone.\n"
            + "Add a no-ROM test for that order if the existing purchase tests do not cover it, and run those tests before the ROM.\n"
            + "Then one planner run from herd_cow1_f627470 as the rules describe. Write snes/harvest/recordings/herd_cow1.json. You may --save-end-state LN_herd_after_cow1 only when the new slot exists and the farmer has left the shop.\n"
            + "passed is true only if you read frames greater than 0, num_cows 0 to 1, cow_slots showing the new slot, wallet down by 5000 from the buy morning, planted_grass at least 32, tilemap not 0x24 at the end of the buy, the next morning's care, happiness not down after the house sleep, ram_writes 0, and mid_run_state_loads 0.\n"
            + "Two tries, then passed false. Put the text id and the cursor byte in blocked if the name screen still does not open."
    ),
    #{ label: "name", capability_mode: "all", output_schema: gate_schema }
);

let halt = "";
if !gate_ok(cow1) {
    halt = "Cow name and raise failed.";
}

phase("Cow2");
let cow2 = ();
if halt == "" {
    let body = "";
    body += "Cow 1 evidence: ";
    body += text_of(cow1.output.evidence);
    body += "\nBuy the second cow before any chicken, from LN_herd_after_cow1. next_livestock returns cow when num_cows is 1 and purchase_block_reason passes. Grass must cover 16 times the animal total after the buy. While num_cows is under 2, do not return chicken. Do not schedule a third cow.\n";
    body += "The same name-and-slot rule applies to this cow. Cap --max-frames 180000. Write snes/harvest/recordings/herd_cow2.json. You may --save-end-state LN_herd_after_cow2 only when the second slot exists and the shop is left.\n";
    body += "passed is true only if you read frames greater than 0, cows 1 to 2, a second initialized slot, wallet down by 5000, cow 1 still present, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "Two tries, then passed false.\n";
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
    body += "\nPlant Year 2 spring with the ordinary planner. Year byte 1, season 0. Potato or turnip only. Tile 0xEB does not count. Year 2 Spring D2 stays an ordinary day. Berry runs stay off when year is not 0. Do not change the cow name path.\n";
    body += "New Year on Year 2 Spring D1 is a forced festival. Use a clean Year 2 spring pin if one exists and was not date-poked. If none is clean, set passed false and name the missing pin. Do not edit the calendar.\n";
    body += "Cap --max-frames 400000 --until-day 6. Write snes/harvest/recordings/y2_spring_plant.json.\n";
    body += "passed is true only if you read frames greater than 0, year 1, season 0, at least one potato or turnip tile other than 0xEB, ram_writes 0, and mid_run_state_loads 0.\n";
    spring = agent(gate_prompt("the year-2 spring planter", body), #{ label: "spring", capability_mode: "all", output_schema: gate_schema });
    if !gate_ok(spring) {
        halt = "Year 2 spring planting receipt failed.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let body = "";
    body += "Read only. Open snes/harvest/recordings/herd_cow1.json, herd_cow2.json, and y2_spring_plant.json.\n";
    body += "cow1 is true only if herd_cow1.json shows cows at least 1, a cow slot, wallet down, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "cow2 is true only if herd_cow2.json shows cows at least 2, wallet down, ram_writes 0, and mid_run_state_loads 0.\n";
    body += "spring is true only if y2_spring_plant.json shows year 1, season 0, a potato or turnip tile other than 0xEB, ram_writes 0, and mid_run_state_loads 0.\n";
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
let path = write_scratch_file("herd_name_report.md", report);
log(report);
complete(#{ path: path, report: report, accepted: halt == "" });
