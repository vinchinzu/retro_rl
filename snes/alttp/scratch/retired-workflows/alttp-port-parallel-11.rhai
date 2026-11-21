let meta = #{
    name: "alttp-port-parallel-11",
    description: "Sword the Kholdstare eyes off the ice, and open the library by door 26",
    when_to_use: "After alttp-port-parallel-10. AP-120 is done. The shell breaks and the sword deals 4.",
    phases: [
        #{ title: "Lanes", detail: "Kholdstare sword, library door" },
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
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite Whatif_Ice_kholdstare, Whatif_Ice_kholdstare_magic, Whatif_Library, Whatif_Hera_moldorm, or recordings/verified_tip_run.json.\n";
    p += "cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes. Read files with read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark done only when the task's own Done check passes.\n";
    p += "Return completed, waiting_on, and blocked.\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Kholdstare fight lane. Claim name is exactly kholdstare-fight. AP-168 is [~] kholdstare-fight. AP-120 is done. Do not edit snes/alttp/bosses/moldorm.py.\n";
ice += "Whatif_Ice_kholdstare_magic has magic 128, a disclosed poke. Eight fire shots take the shell from 64 to 0 and three eyes spawn at ai 0 hp 64. A master sword from the south window deals 4, so one eye went 64 to 60. Offset 0 then dies to falling ice for 16 and eye contact for 32 before $F36C rises. Screen Y 305 latches $0F00 if BG2 leaves the shell. The fight may rewrite BG2 $E2 and $E8 only. Wall rows 27-38 stay unwalked.\n";
ice += "Keep the eight fire shots. Do not stand in the south window while the ice falls, and do not stay in eye contact. Hit for the measured 4 damage and leave. Mark AP-168 done only when the heart is taken on at least 10 of 12 offsets. If offset 0 still dies to ice 16 and eye contact 32, try one different exit from the window. If that also dies the same way, note both poses and leave the claim [~]. Do not block AP-168.\n";
ice += "You may edit snes/alttp/bosses/kholdstare.py, snes/alttp/docs/routes/ice_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-168/. Do not edit walk_physics.py.\n";
jobs.push(#{ prompt: ice, label: "mgr-kholdstare", capability_mode: "all", output_schema: lane_schema });

let book = common();
book += "You are the Book of Mudora lane. Claim name is exactly book-mudora. AP-090 is [~] book-mudora.\n";
book += "Room 0x003D on Whatif_Library is closed. The full 64x64 table has no dash-bit attribute 50, 51, 57, 8E, or 8F. Shelf tiles are attribute 66. The chest at tile (14,48) is KeyA2. A sets tile_action 5 or 3, state stays 0, and $F34E stays 0. Do not warp room 61 again and do not dash screens 0x2B or 0x1B.\n";
book += "z3-json-data names Light World door id 26 for the Library, separate from cave room id 61. From recordings/probe_AP-089/boots.state, trigger door 26 the way an overworld door loads. Read the door table before you boot. If the room that loads is 0x003D, stop and note that. If a different room loads and it has a dash-bit attribute, dash that tile. Mark AP-090 done only when $F34E rises in play, with the room and the ledger frame. Otherwise leave the claim [~]. Do not block AP-090.\n";
book += "You may edit recordings under snes/alttp/recordings/probe_AP-090/. Do not edit boss modules or primitives.py.\n";
jobs.push(#{ prompt: book, label: "mgr-book", capability_mode: "all", output_schema: lane_schema });

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-090, AP-112, AP-120, AP-165, and AP-168. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Eleventh wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
