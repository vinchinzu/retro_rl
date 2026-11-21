let meta = #{
    name: "harvest-marriage",
    description: "Fix the Harvest marriage path: feather stand, Saturday proposal, wife spike, courting, peddler buy, ten-day soak, then one power-on ending",
    when_to_use: "After the marriage audit. A fixture proposal is not a wedding. Stops when a receipt fails twice. Does not edit plan.md or beads.",
    phases: [
        #{ title: "Feather", detail: "shed stand from the tape, then a short probe" },
        #{ title: "Propose", detail: "Saturday, optional, girl-bound stand" },
        #{ title: "Wife", detail: "Eve at the bar, Ann if that misses" },
        #{ title: "Court", detail: "power-on to Summer D1 at 250 hearts" },
        #{ title: "Buy", detail: "peddler text 0x01C3, wallet minus 1000" },
        #{ title: "Soak", detail: "ten married days before the long run" },
        #{ title: "Play", detail: "one power-on ending" },
        #{ title: "Verify", detail: "fail closed on the JSON" },
    ],
};

let gate_schema = #{
    "type": "object",
    "required": ["passed", "evidence", "frames", "report_path", "blocked"],
    "properties": #{
        "passed": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "frames": #{ "type": "integer" },
        "report_path": #{ "type": "string" },
        "blocked": #{ "type": "string" },
        "girl": #{ "type": "string" },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["accepted", "evidence", "frames"],
    "properties": #{
        "accepted": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "frames": #{ "type": "integer" },
        "marriage": #{ "type": "string" },
        "court": #{ "type": "string" },
        "purchase": #{ "type": "string" },
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
    if type_of(r.output.frames) != "integer" { return false; }
    if r.output.frames <= 0 { return false; }
    return true;
}

fn wife_ok(r) {
    if !gate_ok(r) { return false; }
    if r.output.girl != "eve" && r.output.girl != "ann" { return false; }
    return true;
}

fn rules() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Package harvest is snes/harvest, import harvest.\n";
    p += "Read snes/harvest/AGENTS.md and snes/harvest/CONTEXT.md before you edit or boot a ROM. Use read_file and grep on the files you change. Do not answer from this prompt alone.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md or snes/harvest/docs/plan.md. Do not run bd. Do not git add, commit, or push. Do not replace snes/harvest/docs/tasks/rr-20w-run16-defects.md.\n";
    p += "Clean means controller input only. RAM writes forbidden. A mid-run state load is not arrival. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "Do not record a walk a search can already close. Scan snes/harvest/tasks/get_engaged.json before you change a stand or a press.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4. No --video.\n";
    p += "ROM command, from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 <flags>.\n";
    p += "Flags that exist: --state, --power-on, --day-plan, --days, --until-day, --until-season, --until-year, --until-ending, --max-frames, --out, --save-end-state. --until-day stops on that day's morning.\n";
    p += "A window stops when the same farmer pixel repeats for 180 frames. Do not wait out a 48k or 60k day budget. Short probes pass --max-frames. Do not use --max-frames to cut a multi-day receipt short of its day target.\n";
    p += "Two honest tries on a ROM receipt, then stop. Do not weaken a check. Overwrite the one JSON path named in your task. Do not mint a _vN report.\n";
    p += "Re-count a file before you edit it. Soft max is about 1000 lines. day_phase_registry.py and tasks/social.py are at that cap.\n";
    p += "Courting and the feather purchase are new behavior. Put them in snes/harvest/harvest/tasks/proposal.py. Leave ProposalTask and proposal_phase_for_ram in day_phase_registry.py. Do not move those functions into the new file. Do not split social.py. The registry may import one function from proposal.py. If that import would push the registry over 1000 lines, delete dead lines in the registry first.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest <path> -q -m \"not rom\". Do not boot a ROM from the no-ROM suite.\n";
    p += "A passing proposal from Y1_Probe_BuySummerEnd proves the proposal step only. It does not prove courting, the feather purchase, or a power-on wedding.\n";
    return p;
}

fn gate_prompt(role, body) {
    let p = rules();
    p += "You are ";
    p += role;
    p += ".\n";
    p += body;
    p += "\nReturn passed, evidence, frames, report_path, and blocked. passed is true only with a file path you opened and numbers you read from it. frames is an integer greater than 0 from that file. An empty evidence string is a failure. Put the blocker in blocked when passed is false.\n";
    return p;
}

phase("Feather");
let feather_body = "";
feather_body += "Fix the blue-feather shed pickup. The shelf table is SHED_TOOL_SPECS in snes/harvest/harvest/planner/tasks/shed_slots.py. Tool 0x18 is currently inside_stand_px (176, 216) and ShedShelfSpec.inside_face defaults to up.\n";
feather_body += "Read snes/harvest/tasks/get_engaged.json around frame 460 before you edit. The audit measured a pickup at (176, 185), Down+A, held_item 0 to 0x41. The shelf sprite is (176, 192). The row-3 flag is bit 0x80 at shed byte 0x11F02. Stand above the shelf and face down.\n";
feather_body += "If the tape disagrees with those numbers, the tape wins. Say so in evidence.\n";
feather_body += "Set the 0x18 spec to stand (176, 185) and inside_face down. The pickup success check must require selected_tool == 0x18 and the shelf bit clear. Read the task that consumes inside_stand_px and inside_face. Change that check in the file that already owns it. Do not edit day_phase_registry.py.\n";
feather_body += "Extend the existing no-ROM test that already mentions SHED_TOOL_SPECS or tool 0x18. Do not add a new test module if one already covers this spec. Run that test file with the no-ROM pytest line.\n";
feather_body += "Then one short ROM probe. Search saved states offline for a morning house or shed state that already has bit 0x80 set at 0x11F02. Y1_Inside_Shed is a candidate name if the file exists. Do not start from Y1_Probe_BuySummerEnd. That fixture is inside shop 0x1C at 12:05.\n";
feather_body += "Cap the probe with --max-frames 8000. Write snes/harvest/recordings/marriage_feather.json. passed is true only if you read frames > 0, tool_selected 0x18, the shelf bit clear at the end, ram_writes 0, and mid_run_state_loads 0.\n";
feather_body += "If no morning state has the bit, set passed false and name the states you opened. Do not RAM-write the feather onto the shelf. Two tries, then stop.\n";
let feather = agent(
    gate_prompt("the feather-stand fixer", feather_body),
    #{ label: "feather", capability_mode: "all", output_schema: gate_schema }
);

let halt = "";
if !gate_ok(feather) {
    halt = "Feather pickup failed.";
} else {
    log("feather pickup receipt held");
}

phase("Propose");
let propose = ();
if halt == "" {
    let propose_body = "";
    propose_body += "Make PROPOSE match the one verified tape, and stop the shop-exit stall.\n";
    propose_body += "Read proposal_phase_for_ram, _best_girl, _ann_stand, _to_ann, _to_farm_for_feather, and ProposalTask in snes/harvest/harvest/planner/day_phase_registry.py. Also read the get_engaged tape for the proposal press.\n";
    propose_body += "The audit measured one wedding: Summer D4, Saturday, 10:05, town (402, 901), Y+Up. Ann had 853 hearts. marriage_flags went 0 to 2 at frame 2943. Summer D1 is a Wednesday, so Saturday is D4. A --days 2 window cannot reach it. D2 of that load was a storm with no phases.\n";
    propose_body += "_ann_stand is Ann's spring schedule. Weekday 6 is (0x04, 376, 888), which is not the tape stand. _best_girl can return any of the five girls, and the route still walks Ann's stand. Bind the route to the girl you are proposing to. Do not send Eve, Nina, Ellen, or Maria to Ann's spring tiles.\n";
    propose_body += "Schedule PROPOSE on the next Saturday that is not a storm and not a festival, and only before 11:00. That is the only tape-verified slot. Skip other hours instead of failing the day.\n";
    propose_body += "failure_policy is required today, and a miss marks the day bad. Make this phase optional. A miss retries the next eligible Saturday. Cap one attempt at about 12000 frames.\n";
    propose_body += "Y1_Probe_BuySummerEnd starts inside shop 0x1C. _to_farm_for_feather keeps waypoints with pos.y > 280 and has no stall stop, which burned about 62000 frames. Fail that exit when the same pixel repeats for 180 frames. Do not treat a south hack as progress.\n";
    propose_body += "Stay inside day_phase_registry.py for this step. Do not create proposal.py here. Do not change the 0x18 shelf coordinate.\n";
    propose_body += "Add or extend a no-ROM test in the existing day-phase registry tests. Cover Saturday-before-11:00, optional failure, and a non-Ann girl not using _ann_stand. Run that test file.\n";
    propose_body += "Receipt: load Y1_Probe_BuySummerEnd and run until Summer D5 morning. Pass --until-season and --until-day for that morning. Do not pass --days 2. Save the end state as LN_marriage_after_propose only if the wife bit is set. Do not overwrite an existing pin name.\n";
    propose_body += "Write snes/harvest/recordings/marriage_propose.json. passed is true only if you read frames > 0, a wife bit clear at the start and set at the end, ram_writes 0, and mid_run_state_loads 0. This file does not prove courting or the purchase.\n";
    propose_body += "Two tries, then passed false.\n";
    propose = agent(
        gate_prompt("the proposal fixer", propose_body),
        #{ label: "propose", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(propose) {
        halt = "Proposal receipt failed.";
    } else {
        log("proposal receipt held");
    }
}

phase("Wife");
let wife = ();
if halt == "" {
    let wife_body = "";
    wife_body += "Pick the girl. Both Ann and Eve clear 250 hearts on Y1_Probe_BuySummerEnd. One cheap ROM check each, not a search.\n";
    wife_body += "Try Eve at the bar first. Her proposal stand is unmeasured. Read the talk loop and the town map before you send her to a coordinate. One attempt, --max-frames 12000, same-pixel stop at 180 frames. If marriage_flags gains her wife bit, girl is eve.\n";
    wife_body += "If the bar attempt misses, one Ann attempt on the tape slot only: Saturday before 11:00, town (402, 901), Y+Up. If that sets the wife bit, girl is ann.\n";
    wife_body += "Do not walk the other three girls. Do not RAM-write hearts or flags. Do not run --until-ending.\n";
    wife_body += "Write snes/harvest/recordings/marriage_wife.json. Return girl as eve or ann. passed is true only if that file shows frames > 0, the wife bit flip, ram_writes 0, and mid_run_state_loads 0. If both attempts miss, girl is an empty string and passed is false.\n";
    wife = agent(
        gate_prompt("the wife-spike runner", wife_body),
        #{ label: "wife", capability_mode: "all", output_schema: gate_schema }
    );
    if !wife_ok(wife) {
        halt = "Wife spike did not pick Eve or Ann.";
    } else {
        log("wife spike picked a girl");
    }
}

let girl = "";
if wife != () && wife.output != () && type_of(wife.output.girl) == "string" {
    girl = wife.output.girl;
}

phase("Court");
let court = ();
if halt == "" {
    let court_body = "";
    court_body += "Add native courting for ";
    court_body += girl;
    court_body += " to the day planner that a power-on run actually calls. A named --day-plan is not enough. The Eve talk loop exists as EveTalkLoopTask and is only scheduled when someone passes that day plan.\n";
    court_body += "Read harvest/tasks/eve_loop_task.py and the phase builder before you edit. The loop is capped at 120 points a day. Ann has no courting executor. A gift is about +6 hearts, so 250 hearts is about 40 gifts. Read the heart code or the tape before you hard-code either number. If the code disagrees, the code wins.\n";
    court_body += "Put the new courting phase in snes/harvest/harvest/tasks/proposal.py. Call it from the registry with one import if the planner does not already see it. Do not move ProposalTask. Do not edit social.py unless the gift or talk primitive already lives there and you are calling it.\n";
    court_body += "Court only ";
    court_body += girl;
    court_body += ". Do not raise the other girls.\n";
    court_body += "Receipt: power-on until Summer D1 morning. Use --power-on --until-season 1 --until-day 1. Write snes/harvest/recordings/marriage_court.json. passed is true only if you read frames > 0, that girl's hearts >= 250, ram_writes 0, and mid_run_state_loads 0. Hearts at the power-on start are not 250. The audit scored the power-on ending at 5 of 155 girl points.\n";
    court_body += "Stop a day when the same pixel repeats for 180 frames. Two tries, then passed false. Do not run --until-ending.\n";
    court = agent(
        gate_prompt("the courting author", court_body),
        #{ label: "court", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(court) {
        halt = "Courting receipt failed.";
    } else {
        log("courting receipt held");
    }
}

phase("Buy");
let bought = ();
if halt == "" {
    let buy_body = "";
    buy_body += "Buy the blue feather in play. Text 0x01C3 is Text_Peddler_BlueFeather1 at DATA16_B8892A. The price is 1000G. No current code says the day, map, or hour.\n";
    buy_body += "Decode the script that loads 0x01C3. Start in snes/harvest/HM-Decomp and the RAM notes. Read snes/harvest/harvest/planner/tasks/inventory_exit.py before you reuse the farm-visitor wait. That wait dismisses a visitor who already owns the farmer. It is not proof the peddler is that visitor. livestock_econ.py has a year-2 visitor money floor. Do not treat that floor as the feather sale.\n";
    buy_body += "Put the purchase in snes/harvest/harvest/tasks/proposal.py next to the courting code. Do not delete the courting functions. Wire it into the same power-on day planner. Do not move ProposalTask.\n";
    buy_body += "Receipt: wallet down by 1000 and the shed shelf bit 0x80 at 0x11F02 set, from a state that had the bit clear and at least 1000G. Controller input only. Write snes/harvest/recordings/marriage_feather_buy.json.\n";
    buy_body += "passed is true only if you read frames > 0, wallet delta -1000, the shelf bit clear then set, ram_writes 0, and mid_run_state_loads 0. A menu, or a return to the origin map, is not a purchase.\n";
    buy_body += "Bound the wait with the 180-frame same-pixel stop. Two tries, then passed false. Do not run --until-ending.\n";
    bought = agent(
        gate_prompt("the feather buyer", buy_body),
        #{ label: "buy", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(bought) {
        halt = "Feather purchase failed.";
    } else {
        log("feather purchase receipt held");
    }
}

phase("Soak");
let soaked = ();
if halt == "" {
    let soak_body = "";
    soak_body += "Run about ten married days before anyone starts the year-3 ending. Load LN_marriage_after_propose. If that pin is missing or the wife bit is clear, set passed false. Do not borrow a what-if married pin.\n";
    soak_body += "Use --days 10 or an --until-day that is ten mornings later. Write snes/harvest/recordings/marriage_soak.json.\n";
    soak_body += "The get_engaged tape rewinds the clock to 07:05 after scene 0x2D. The wife's hour-18 script can sit on an A-wait. House level 2 changes the bed. Read snes/harvest/debug_alignment/l2_house_wife_bed/ and the sleep task before you call a stall a success.\n";
    soak_body += "If sleep never reaches NightReset, fix that once in the file that owns the bed press, then retry the ten days once. Same-pixel stop at 180 frames still applies. Do not raise the day budget to hide the wait.\n";
    soak_body += "passed is true only if you read frames > 0, at least ten overnights, the wife bit still set, ram_writes 0, and mid_run_state_loads 0. Do not run --until-ending.\n";
    soaked = agent(
        gate_prompt("the married-days runner", soak_body),
        #{ label: "soak", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(soaked) {
        halt = "Married-days soak failed.";
    } else {
        log("ten-day soak held");
    }
}

phase("Play");
let played = ();
if halt == "" {
    let play_body = "";
    play_body += "Courting evidence: ";
    play_body += text_of(court.output.evidence);
    play_body += "\nPurchase evidence: ";
    play_body += text_of(bought.output.evidence);
    play_body += "\nSoak evidence: ";
    play_body += text_of(soaked.output.evidence);
    play_body += "\nOpen snes/harvest/recordings/marriage_court.json and snes/harvest/recordings/marriage_feather_buy.json yourself. If either file lacks ram_writes 0, do not start the ending. marriage_propose.json is not enough.\n";
    play_body += "Run exactly one power-on ending from the repo root:\n";
    play_body += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 --power-on --until-ending --out snes/harvest/recordings/full_year/power_on_marriage.json\n";
    play_body += "Let it finish. One emulator. Do not start a second. This is the long run, on the order of 35 minutes, and only this phase may start it.\n";
    play_body += "passed is true only if you opened that JSON and it shows frames > 0, ending finished, a wife bit in marriage_flags, ram_writes 0, and mid_run_state_loads 0. The wife bit has to come from the courting and purchase path, not from a RAM write.\n";
    play_body += "If the process dies, one retry after you fix the crash. Then passed false. report_path is snes/harvest/recordings/full_year/power_on_marriage.json.\n";
    played = agent(
        gate_prompt("the ending runner", play_body),
        #{ label: "play", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(played) {
        halt = "Power-on ending failed the marriage checks.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let verify_body = "";
    verify_body += "You are the skeptic. Read only. Open these three files with read_file. Do not trust the runner text.\n";
    verify_body += "snes/harvest/recordings/full_year/power_on_marriage.json\n";
    verify_body += "snes/harvest/recordings/marriage_court.json\n";
    verify_body += "snes/harvest/recordings/marriage_feather_buy.json\n";
    verify_body += "Runner evidence, untrusted: ";
    verify_body += text_of(played.output.evidence);
    verify_body += "\nSet accepted true only if the ending file shows power_on, until_ending, frames greater than 0, ending finished, a wife bit in marriage_flags, ram_writes 0, and mid_run_state_loads 0.\n";
    verify_body += "Also require the court file to show the chosen girl's hearts at or above 250 with ram_writes 0, and the purchase file to show wallet delta -1000 and the shelf bit set with ram_writes 0.\n";
    verify_body += "marriage_propose.json cannot satisfy this check. If a field is absent, accepted is false. Put the numbers you read into evidence, marriage, court, and purchase. frames is the ending integer, or 0 when you reject.\n";
    verdict = agent(
        verify_body,
        #{ label: "verify", capability_mode: "read-only", output_schema: verdict_schema }
    );
}

let accepted = false;
let frame_text = "0";
if halt == "" && verdict != () && verdict.success == true && verdict.output != () && verdict.output.accepted == true {
    if type_of(verdict.output.evidence) == "string" && verdict.output.evidence != "" && type_of(verdict.output.frames) == "integer" && verdict.output.frames > 0 {
        accepted = true;
        frame_text = verdict.output.frames.to_string();
    } else {
        halt = "Verifier accepted without frames or evidence.";
    }
} else if halt == "" {
    halt = "Verifier rejected the marriage ending.";
}

let summary = "";
if accepted {
    summary = "Marriage path accepted. Power-on ending has a wife bit, courting and the 1000G feather buy are on disk, and the run is ";
    summary += frame_text;
    summary += " frames. Girl: ";
    summary += girl;
    summary += ". ";
    summary += text_of(verdict.output.evidence);
} else {
    summary = "Marriage path stopped. ";
    summary += halt;
    summary += " Girl: ";
    summary += girl;
    summary += ".";
    if feather != () && feather.output != () && text_of(feather.output.blocked) != "" {
        summary += " Feather: ";
        summary += text_of(feather.output.blocked);
    }
    if propose != () && propose.output != () && text_of(propose.output.blocked) != "" {
        summary += " Propose: ";
        summary += text_of(propose.output.blocked);
    }
    if wife != () && wife.output != () && text_of(wife.output.blocked) != "" {
        summary += " Wife: ";
        summary += text_of(wife.output.blocked);
    }
    if court != () && court.output != () && text_of(court.output.blocked) != "" {
        summary += " Court: ";
        summary += text_of(court.output.blocked);
    }
    if bought != () && bought.output != () && text_of(bought.output.blocked) != "" {
        summary += " Buy: ";
        summary += text_of(bought.output.blocked);
    }
    if soaked != () && soaked.output != () && text_of(soaked.output.blocked) != "" {
        summary += " Soak: ";
        summary += text_of(soaked.output.blocked);
    }
    if played != () && played.output != () && text_of(played.output.blocked) != "" {
        summary += " Play: ";
        summary += text_of(played.output.blocked);
    }
}

let report_path = write_scratch_file("harvest-marriage.md", summary);
complete(#{
    summary: summary,
    accepted: accepted,
    girl: girl,
    frames: frame_text,
    path: report_path,
});
