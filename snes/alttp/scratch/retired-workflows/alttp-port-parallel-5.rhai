let meta = #{
    name: "alttp-port-parallel-5",
    description: "Finish Moldorm from the 3 passing offsets, survive Lanmolas, and read the Eastern pendant byte",
    when_to_use: "After alttp-port-parallel-4. AP-104 and AP-111 are done. AP-120 is claimed.",
    phases: [
        #{ title: "Lanes", detail: "Moldorm, Lanmolas, pendant byte" },
        #{ title: "Report", detail: "tick counts" },
    ],
};

let lane_schema = #{
    "type": "object",
    "required": ["completed", "waiting_on", "blocked"],
    "properties": #{
        "completed": #{ "type": "array", "items": #{ "type": "string" } },
        "waiting_on": #{ "type": "string" },
        "blocked": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["summary", "open_count", "done_count", "blocked_count"],
    "properties": #{
        "summary": #{ "type": "string" },
        "open_count": #{ "type": "integer" },
        "done_count": #{ "type": "integer" },
        "blocked_count": #{ "type": "integer" },
    },
};

fn common() {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md.\n";
    p += "Pins in this wave are isolated. They are not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, Whatif_Eastern_armos, Whatif_Desert_lanmolas, Whatif_Hera_moldorm, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let hera = common();
hera += "You are the Moldorm fight lane. Claim name is exactly moldorm-fight, because AP-120 is already [~] moldorm-fight. Do not touch AP-118 or AP-119.\n";
hera += "Closed fact: on Whatif_Hera_moldorm, offsets 0, 2, and 4 take the heart (frames 982, 818, 817, hp_lost 0). Offsets 1, 3, 5, 6, 7, 8, 9, 10, and 11 hit the 7500-frame cap. Walked rel=(-4,0) facing UP whiffs. Do not rerun that same policy and do not repeat the whiff.\n";
hera += "Read snes/alttp/bosses/moldorm.py. Change the fight so the nine capped initial phases also take the heart. Score all 12 offsets. Mark AP-120 done only when the heart is taken on at least 10 of 12. If you are still under 10, add a note with the new pass list and leave the claim [~] under moldorm-fight. Do not block AP-120.\n";
hera += "You may edit snes/alttp/bosses/moldorm.py, snes/alttp/docs/routes/hera_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-119/. Do not edit armos_knights.py or lanmolas.py.\n";
jobs.push(#{ prompt: hera, label: "mgr-moldorm", capability_mode: "all", output_schema: lane_schema });

let desert = common();
desert += "You are the Lanmolas fight lane. AP-111 is done. AP-112 is [!]. Whatif_Desert_lanmolas is indoors in room 0x33 with three sprite 0x54. Sword drops $0E50 by 2 while $0D80 is 2 or 3 at distance 20-28. Indoor boulder 0xC2 speed is its velocity byte over 16.\n";
desert += "Closed fight: offset 0 of bosses/lanmolas:fight died at frame 1117, hp 24 to 0, hits lanmolas:3, strikes 0. An earlier sample died at frame 6362 after boulder:3 and lanmolas:1. Do not repeat a policy that stands in the boulders.\n";
desert += "Read snes/alttp/bosses/lanmolas.py. New policy: stay out of 0xC2 and swing only in the measured window. Claim AP-112. Mark it done only when the heart is taken on at least 10 of 12 offsets. If you die again, block AP-112 with the new frame, hp, and hit list.\n";
desert += "You may edit snes/alttp/bosses/lanmolas.py, snes/alttp/docs/routes/desert_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-111/. Do not edit moldorm.py.\n";
jobs.push(#{ prompt: desert, label: "mgr-lanmolas", capability_mode: "all", output_schema: lane_schema });

let eastern = common();
eastern += "You are the Eastern pendant lane. AP-103 and AP-104 are done. The 12/12 fight on Whatif_Eastern_armos took the heart container, max HP 0x18 to 0x20. Do not refight the knights and do not edit armos_knights.py.\n";
eastern += "AP-105 needs AP-086, which is blocked. Do not claim AP-105.\n";
eastern += "Read AP-106. Its acceptance is the pendant bit, candidate $F374, rising in play. Load the existing clear only far enough to read $F374 before and after the prize. Do not write that byte. If it rises and the Done line is only that byte, mark AP-106 done and name the pin. If the Done line also requires the Sanctuary chain, leave AP-106 open and note the before and after values.\n";
eastern += "You may edit snes/alttp/docs/routes/eastern_ROUTE.md and a scratch script under snes/alttp/recordings/probe_AP-106/. Do not edit boss modules.\n";
jobs.push(#{ prompt: eastern, label: "mgr-pendant", capability_mode: "all", output_schema: lane_schema });

let wave = parallel(jobs);
let n = 0;
if wave != () {
    for slot in wave {
        if slot != () && slot.success {
            n += 1;
        }
    }
}
log("lanes returned " + n.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-106, AP-112, AP-120, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Fifth wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
