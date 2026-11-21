let meta = #{
    name: "harvest-finish",
    description: "Finish the Harvest Moon power-on play: seasons, one cow, one chicken, marriage, credits score, frame count",
    when_to_use: "Drive the Finish rung. Not ranch-master 999. Stops when a receipt fails twice.",
    phases: [
        #{ title: "Audit", detail: "read the Finish gaps" },
        #{ title: "Bugs", detail: "no-ROM suite, one fix pass" },
        #{ title: "Animals", detail: "one cow and one chicken, frames counted" },
        #{ title: "Marriage", detail: "wife bit from controller input" },
        #{ title: "Play", detail: "one power-on run through credits" },
        #{ title: "Verify", detail: "fail closed on the JSON" },
    ],
};

let audit_schema = #{
    "type": "object",
    "required": ["evidence", "animals_proven", "marriage_proven", "seasons_proven", "score_proven"],
    "properties": #{
        "evidence": #{ "type": "string" },
        "animals_proven": #{ "type": "boolean" },
        "marriage_proven": #{ "type": "boolean" },
        "seasons_proven": #{ "type": "boolean" },
        "score_proven": #{ "type": "boolean" },
        "frame_count": #{ "type": "string" },
        "gaps": #{ "type": "string" },
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
    "required": ["accepted", "evidence", "frames"],
    "properties": #{
        "accepted": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "frames": #{ "type": "integer" },
        "seasons": #{ "type": "string" },
        "marriage": #{ "type": "string" },
        "chickens": #{ "type": "string" },
        "cows": #{ "type": "string" },
        "score": #{ "type": "string" },
    },
};

fn truth(v) {
    if v == true { return true; }
    return false;
}

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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Package harvest lives in snes/harvest.\n";
    p += "Read snes/harvest/CONTEXT.md, snes/harvest/docs/plan.md, snes/harvest/docs/STATUS.md, snes/harvest/docs/livestock_purchase_plan.md, and snes/harvest/AGENTS.md before you edit or boot a ROM.\n";
    p += "This campaign is Finish: a Clean power-on play through Year 3 credits that shows a score, all four seasons, at least one cow, at least one chicken, and a marriage. It is not ranch-master 999. Do not open the perfect-rancher beads and do not chase twelve animals or exactly ten berries.\n";
    p += "Do not edit snes/harvest/docs/STATUS.md. Do not run bd. Do not git add, commit, or push.\n";
    p += "Clean: controller input only. RAM writes forbidden. A mid-run state load does not count as arrival. Do not start Day 2 from Y1_D2_Morning_After_D1.\n";
    p += "Do not record a walk a search can already close. A shop menu, or a cross-map return to the origin, is not a buy. A buy needs the shop tilemap, a wallet change, and an animal-count change. A cow buy also needs a newly initialized slot.\n";
    p += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen. One emulator. Do not kill other Python processes. No MP4.\n";
    p += "Re-count a file before editing it. Soft max is about 1000 lines. multi_day_planner.py and tasks/social.py are at that cap: replace inside the file that already owns the behavior, or fold into the composer it already calls. No new sibling module.\n";
    p += "Two honest tries on a ROM receipt, then stop. Do not weaken a check. Do not mint a _vN report. Overwrite the one JSON path named in your task.\n";
    p += "Pytest from the repo root: QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest <path> -q. Mark ROM tests with the existing rom marker. Do not boot a ROM from the no-ROM suite.\n";
    return p;
}

fn gate_prompt(role, body) {
    let p = rules();
    p += "You are ";
    p += role;
    p += ".\n";
    p += body;
    p += "\nReturn passed, evidence, frames, report_path, and blocked. passed is true only with a file path you opened and numbers you read from it. An empty evidence string is a failure.\n";
    return p;
}

phase("Audit");
let audit = agent(
    rules() + "You are the auditor. Read only. Use read_file and grep.\n"
        + "Decide whether any existing power-on --until-ending JSON already proves Finish.\n"
        + "animals_proven is true only if that same JSON shows num_cows >= 1 and num_chickens >= 1, with wallet and slot evidence, ram writes 0, mid-run loads 0.\n"
        + "marriage_proven is true only if that same JSON shows a wife bit in marriage_flags from controller play, not a what-if pin.\n"
        + "seasons_proven is true only if that same JSON's day journal contains season 0, 1, 2, and 3 and the ending finished.\n"
        + "score_proven is true only if that same JSON records the credits score as a number.\n"
        + "The 2175350-frame working-farm credits run plants Spring and Summer and sleeps early in Fall and Winter. It is not animals_proven and not marriage_proven.\n"
        + "Pin receipts under recordings/rr-bu62.28-*.json are not power-on proof. Set the booleans false unless you opened a power-on until-ending file that contains the fields.\n"
        + "Return evidence, animals_proven, marriage_proven, seasons_proven, score_proven, frame_count, and gaps. Quote paths and numbers you read.",
    #{ label: "audit", capability_mode: "read-only", output_schema: audit_schema }
);

let halt = "";
let audit_json = "";
if audit == () || audit.success != true || audit.output == () || type_of(audit.output.evidence) != "string" || audit.output.evidence == "" {
    halt = "Audit returned no evidence.";
} else {
    audit_json = json_encode(audit.output);
    log("audit gaps recorded");
}

phase("Bugs");
let bugs = ();
if halt == "" {
    let bug_body = "";
    bug_body += "Audit JSON follows. Treat it as a hint, then check the tests yourself.\n";
    bug_body += audit_json;
    bug_body += "\nRun QT_QPA_PLATFORM=offscreen RETRO_RL_TEST_TIER=game-no-rom uv run pytest snes/harvest/tests -m \"not rom\" -q from the repo root.\n";
    bug_body += "If it is green, do not edit, set passed true, and put the pass count in evidence.\n";
    bug_body += "If it is red, fix only the failures in the files that own them. Then run those tests again, and the full no-ROM harvest suite once more.\n";
    bug_body += "One fix pass. If the suite is still red, set passed false and name the failing node ids in blocked. Do not boot a ROM.\n";
    bugs = agent(
        gate_prompt("the bug fixer", bug_body),
        #{ label: "bugs", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(bugs) {
        halt = "No-ROM suite is still red.";
    }
}

phase("Animals");
let animals = ();
if halt == "" {
    if truth(audit.output.animals_proven) {
        log("animals already on a power-on JSON");
        animals = #{ success: true, output: #{ passed: true, evidence: audit.output.evidence, frames: 0, report_path: "", blocked: "" } };
    } else {
        let animal_body = "";
        animal_body += "Audit JSON follows.\n";
        animal_body += audit_json;
        animal_body += "\nMake the live power-on day planner buy one cow and one chicken. Read the scheduler before you change it. The shop gate is planted grass, open weekday, delivery weather, wallet, and a new cow slot after naming.\n";
        animal_body += "Prove it with one bounded ROM window from a real predecessor the planner can reach. Command shape, from the repo root: HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 <flags>.\n";
        animal_body += "Write snes/harvest/recordings/finish_animals.json. passed is true only if you read frames > 0, shop tilemap, wallet down, chickens up by at least 1, cows up by at least 1, a new cow slot, clean_run ram writes 0, mid-run loads 0.\n";
        animal_body += "Do not run --until-ending. Do not RAM-poke gold, grass, hearts, or animal slots. Two tries, then passed false.\n";
        animals = agent(
            gate_prompt("the livestock prover", animal_body),
            #{ label: "animals", capability_mode: "all", output_schema: gate_schema }
        );
    }
    if !gate_ok(animals) {
        halt = "Cow and chicken receipt failed.";
    }
}

phase("Marriage");
let married = ();
if halt == "" {
    if truth(audit.output.marriage_proven) {
        log("marriage already on a power-on JSON");
        married = #{ success: true, output: #{ passed: true, evidence: audit.output.evidence, frames: 0, report_path: "", blocked: "" } };
    } else {
        let marry_body = "";
        marry_body += "Audit JSON follows.\n";
        marry_body += audit_json;
        marry_body += "\nWire the proposal into the same day planner the power-on run uses, if it is not already there. Read harvest/tasks/social.py and the planner before editing. social.py is at the line cap.\n";
        marry_body += "Prove the wife bit with controller input. A poison-mushroom heart drop on a married what-if pin is not a wedding. A RAM write of marriage_flags is not a wedding.\n";
        marry_body += "Use the narrowest run that can show the bit flip. Save the predecessor with the existing --save-end-state flag when you need one. Write snes/harvest/recordings/finish_marriage.json.\n";
        marry_body += "passed is true only if you read frames > 0, marriage_flags wife bit set at the end, and it was clear at the start, with ram writes 0 and mid-run loads 0.\n";
        marry_body += "Do not run --until-ending. Two tries, then passed false. Put the blocker in blocked.\n";
        married = agent(
            gate_prompt("the marriage prover", marry_body),
            #{ label: "marriage", capability_mode: "all", output_schema: gate_schema }
        );
    }
    if !gate_ok(married) {
        halt = "Marriage receipt failed.";
    }
}

phase("Play");
let played = ();
if halt == "" {
    let play_body = "";
    play_body += "Animals evidence: ";
    play_body += text_of(animals.output.evidence);
    play_body += "\nMarriage evidence: ";
    play_body += text_of(married.output.evidence);
    play_body += "\nBefore the long run, read harvest/scripts/run_to_day2.py. The report dict must record end num_cows, num_chickens, marriage_flags, and a numeric credits score. Add those keys to the existing report if they are missing. The file is under the line cap. Do not add a new script.\n";
    play_body += "Then run exactly one power-on ending from the repo root:\n";
    play_body += "HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy QT_QPA_PLATFORM=offscreen uv run python -m harvest.scripts.run_to_day2 --power-on --until-ending --save-season-pins --out snes/harvest/recordings/full_year/power_on_finish.json\n";
    play_body += "This run is long. Let it finish. Do not start a second emulator. Do not treat 2175350 frames as a fail line. Record the frames the JSON actually contains.\n";
    play_body += "passed is true only if you opened that JSON and it shows frames > 0, ending.finished true, day journal seasons 0, 1, 2, and 3, num_cows >= 1, num_chickens >= 1, a wife bit in marriage_flags, a numeric score, clean_run ram writes 0, and mid_run_state_loads 0.\n";
    play_body += "If the run dies, one retry after you fix the crash cause. Then passed false. report_path is snes/harvest/recordings/full_year/power_on_finish.json.\n";
    played = agent(
        gate_prompt("the full-play runner", play_body),
        #{ label: "play", capability_mode: "all", output_schema: gate_schema }
    );
    if !gate_ok(played) {
        halt = "Power-on ending run failed the Finish checks.";
    }
}

phase("Verify");
let verdict = ();
if halt == "" {
    let verify_body = "";
    verify_body += "You are the skeptic. Read only. Open snes/harvest/recordings/full_year/power_on_finish.json with read_file. Do not trust the runner's summary.\n";
    verify_body += "Runner evidence, untrusted: ";
    verify_body += text_of(played.output.evidence);
    verify_body += "\nSet accepted true only if the file shows all of these: power_on, until_ending, frames as an integer greater than 0, ending.finished true, day-journal dates covering season 0, 1, 2, and 3, num_cows >= 1, num_chickens >= 1, marriage_flags with a wife bit, a numeric credits score, clean_run ram writes 0, mid_run_state_loads 0.\n";
    verify_body += "If a field is absent, accepted is false. Put the numbers you read into evidence, seasons, marriage, chickens, cows, and score. frames is the integer from the JSON, or 0 when you reject.\n";
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
    halt = "Verifier rejected the finish JSON.";
}

let summary = "";
if accepted {
    summary = "Finish accepted. Power-on credits JSON has seasons, a cow, a chicken, a marriage, a score, and ";
    summary += frame_text;
    summary += " frames. ";
    summary += text_of(verdict.output.evidence);
} else {
    summary = "Finish stopped. ";
    summary += halt;
    if animals != () && animals.output != () {
        summary += " Animals: ";
        summary += text_of(animals.output.blocked);
    }
    if married != () && married.output != () {
        summary += " Marriage: ";
        summary += text_of(married.output.blocked);
    }
    if played != () && played.output != () {
        summary += " Play: ";
        summary += text_of(played.output.blocked);
    }
}

let report_path = write_scratch_file("harvest-finish.md", summary);
complete(#{ summary: summary, accepted: accepted, frames: frame_text, path: report_path });
