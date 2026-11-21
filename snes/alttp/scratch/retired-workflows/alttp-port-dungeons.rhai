let meta = #{
    name: "alttp-port-dungeons",
    description: "Work every ALttP dungeon from its start pin, skipping closed probes",
    when_to_use: "Finish the dungeon autoplay from indoor start states. Do not repeat a blocked big-key or boss probe.",
    phases: [
        #{ title: "Pins", detail: "missing dungeon-start pins only" },
        #{ title: "Dungeons", detail: "twelve managers from their start pins" },
        #{ title: "Report", detail: "tick counts" },
    ],
};

let pin_schema = #{
    "type": "object",
    "required": ["pins_ready", "missing", "note"],
    "properties": #{
        "pins_ready": #{ "type": "array", "items": #{ "type": "string" } },
        "missing": #{ "type": "array", "items": #{ "type": "string" } },
        "note": #{ "type": "string" },
    },
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

fn mgr_prompt(name, slug, lo, hi, entry, leave, open_ids, closed, route, yaml, boss, pin_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are one dungeon manager. Agent name ";
    p += name;
    p += ". Dungeon slug ";
    p += slug;
    p += " only. Tasks ";
    p += lo;
    p += " through ";
    p += hi;
    p += ".\n";
    p += "Before any tick, export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md and test -f that file. The claim script's default path retro_rl does not exist. Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not edit the claim script. Do not run bd. Do not git add, commit, or checkout. Do not revert unrelated dirty files. Do not edit docs/STATUS.md.\n";
    p += "Read snes/alttp/AGENTS.md, the dungeon template in snes/alttp/docs/routes/DUNGEON_TEMPLATE.md, and only your own headings in PORT_PLAN.md. Do not read other dungeons.\n";
    p += "Owner ruling 2026-09-30: a generated start pin with disclosed inventory RAM writes is enough to work this dungeon. The pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += "Start pin: prefer Whatif_";
    p += entry;
    p += "_parallel when that file exists, else ";
    p += entry;
    p += ". Pin directory: snes/alttp/custom_integrations/Zelda3-Snes/. The pin report JSON is below. If neither pin exists on disk, set waiting_on to your first open id, leave blocked empty, and do not boot a ROM.\n";
    p += "Pin report JSON: ";
    p += pin_note;
    p += "\n";
    p += "Leave these ids alone. Do not claim them, do not mark them done, and do not repeat their probes: ";
    p += leave;
    p += ".\n";
    p += "Snapshot of ids that were open when this wave started: ";
    p += open_ids;
    p += ". Re-read each heading before you touch it. The file wins over this snapshot.\n";
    p += "Closed probes, do not repeat: ";
    p += closed;
    p += "\n";
    p += "Skip [x]. Skip [!] and skip [~] owned by another agent. Do not stop the lane on an earlier blocked id. Work the next open heading whose own Done line can be attempted from the start pin. A Needs line that names an overworld task does not forbid an interior step. If the Done line is the overworld approach or the continuous chain, and that Needs id is not [x], leave the task open, record it as waiting_on, and do not walk the overworld.\n";
    p += "A step is done only when its own Done check passes. Two honest tries, then block that step and continue. After two steps you yourself blocked this sitting, stop the lane. Do not weaken a Done check. If the Done line requires the item to rise in play, a poke does not pass it.\n";
    p += "Boss step 6 needs at least 10 of 12 RNG offsets. Hint lines are not authority. z3-json-data and a live scan win. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not commit refs and do not download.\n";
    p += "You may edit only these paths: ";
    p += route;
    p += ", ";
    p += yaml;
    p += ", ";
    p += boss;
    p += ", maps/room_*.json for rooms your own tasks name, scratch/probe_ap files whose id is inside your range, and recordings under snes/alttp/recordings/probe_";
    p += lo;
    p += "/. Re-read a file immediately before you edit it. Yaml edits are hint fixes only. Do not edit dungeon/bosses/__init__.py.\n";
    p += "Do not edit make_entry_pin.py, whatif_pin.py, hops.py, primitives.py, ram.py, nav.py, tilemap.py, paths.py, walk_physics.py, AGENTS.md, docs/STATUS.md, docs/ASSIST_CONTRACT.md, or another dungeon's files. If a task needs a new shared primitive, set waiting_on to that task id, write the needed change in your route doc, and continue to a later step that does not need it.\n";
    p += "Commands that boot the ROM use cwd snes/ and SDL_VIDEODRIVER=dummy. One emulator in your process. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Soft max about 1000 lines. Geometry stays in maps json. Do not copy approach xy into Python.\n";
    p += "Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed (AP ids you marked [x]), waiting_on (one id or empty), blocked (one id you marked [!] this sitting, or empty).\n";
    return p;
}

phase("Pins");
let pin_prompt = "";
pin_prompt += "Repo: /home/v/01_projects/11_games/retro. You are mgr-pins, the only agent who may create dungeon-start pins. You do not implement dungeon steps.\n";
pin_prompt += "Export ALTTP_PORT_PLAN=/home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md before any tick. Ticks only through python3 /home/v/.grok/alttp_port_claim.py. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md. Do not edit the claim script.\n";
pin_prompt += "Owner ruling 2026-09-30: a start pin may use disclosed RAM writes for inventory, keys, magic, and ability bits. Pins are development only. Provenance must say poke. Never write $F3CC. Do not call a pin natural or continuous. Do not overwrite recordings/verified_tip_run.json.\n";
pin_prompt += "Do not retry closed measurements: Desert ability $F379=0xFC dash at sprite 0x3B; Eastern magic 128 on the big-key chest; Hera basement wall row 52 cols 27-36; another writer of bit 0x04 of $7EF379; A on the Ice room 0x1F chest; the Swamp north door from (4344,1336); the Thieves lift RIGHT from (6088,6903).\n";
pin_prompt += "Do not mark AP-101, AP-109, AP-110, AP-112, AP-117, AP-118, AP-139, AP-155, or AP-165 done. Do not claim AP-113 or AP-168.\n";
pin_prompt += "Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, PalaceOfDarknessEntry, SwampPalaceEntry, ThievesTownEntry, HyruleCastleGrounds, Scan_* pins, Play_* pins, or any existing Whatif_* pin.\n";
pin_prompt += "Read snes/alttp/scripts/make_entry_pin.py and snes/alttp/scripts/whatif_pin.py before you edit or boot. cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator at a time. Do not kill other Python processes.\n";
pin_prompt += "Pin directory is snes/alttp/custom_integrations/Zelda3-Snes/. These indoor pins already exist: Whatif_EasternPalaceEntry_parallel, Whatif_DesertPalaceEntry_parallel, Whatif_HeraEntry_parallel, Whatif_IcePalaceEntry_parallel, PalaceOfDarknessEntry, SwampPalaceEntry, ThievesTownEntry. Confirm each with pin_probe (room and module only). Do not rebuild them.\n";
pin_prompt += "Missing slugs: castle_tower, skull, mire, turtle_rock, gt. make_entry_pin.py ENTRIES currently has only eastern and ice. For each missing slug, search snes/alttp/refs/z3-json-data if present, else snes/alttp/z3-json-data, for an entrance node whose numeric id you can quote. Also read snes/alttp/dungeon/route_data/<slug>.yaml. If you can quote an entrance id, add one EntrySpec, run --print-kit, then one rebuild under the pin name the tool already uses. If the room guard refuses, save under a new name the tool allows. If you cannot quote an entrance id, skip that slug. Do not invent an id. Do not warp a room with submodule 0x15 as a substitute entry pin.\n";
pin_prompt += "If PalaceOfDarkness, Swamp, or Thieves has no Whatif_<Entry>_parallel yet, copy the existing Entry pin with whatif_pin.py. Write only inventory fields a later open step needs, and only fields the tool already names. Refuse $F3CC. Do not clobber the source pin.\n";
pin_prompt += "Verify every pin you name with pin_probe: room, module, and any byte you wrote. Use read_file, grep, and the shell. Do not answer from memory.\n";
pin_prompt += "Return pins_ready (pin names pin_probe showed indoors), missing (slugs with no indoor pin), and note (one paragraph: pin name per slug, and which bytes were writes).\n";
let pins = agent(pin_prompt, #{
    label: "mgr-pins",
    capability_mode: "all",
    output_schema: pin_schema,
});
let pin_note = "{}";
if pins == () || pins.success != true {
    log("pin manager returned no success; dungeon managers still check the disk");
} else if type_of(pins.output) == "map" {
    pin_note = json_encode(pins.output);
    log("pin manager returned a pin report");
}

phase("Dungeons");
let dungeons = [
    #{ name: "Eastern Palace", slug: "eastern", lo: "AP-099", hi: "AP-106", entry: "EasternPalaceEntry", leave: "AP-101", open_ids: "AP-105", closed: "AP-105 is the overworld approach under rr-eog9. That bead is stuck on screen 0x1D at (2664,1544). Do not repeat the north-lip walk and do not start the next bead. If the Done line is that approach, leave AP-105 open and do not boot.", route: "snes/alttp/docs/routes/eastern_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/eastern.yaml", boss: "snes/alttp/dungeon/bosses/armos_knights.py" },
    #{ name: "Desert Palace", slug: "desert", lo: "AP-107", hi: "AP-114", entry: "DesertPalaceEntry", leave: "AP-109 AP-110 AP-112 AP-113", open_ids: "AP-114", closed: "AP-113 is claimed by desert-tablet. Do not steal it. Do not retry the ability-FC dash, A at sprite 0x3B, the glove rock at (704,3168), the Lanmolas boulder policies, or the Desert plaque dashes. If AP-114 needs the boss policy or the tablet, leave it open and do not boot.", route: "snes/alttp/docs/routes/desert_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/desert.yaml", boss: "snes/alttp/dungeon/bosses/lanmolas.py" },
    #{ name: "Tower of Hera", slug: "hera", lo: "AP-115", hi: "AP-122", entry: "HeraEntry", leave: "AP-117 AP-118", open_ids: "AP-121 AP-122", closed: "Do not retry the basement wall on row 52 or the pearl chest. AP-121 Needs AP-114 and AP-091. If those are not [x], do not walk the approach. If neither open step can pass from the start pin, leave them open and do not boot.", route: "snes/alttp/docs/routes/hera_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/hera.yaml", boss: "snes/alttp/dungeon/bosses/moldorm.py" },
    #{ name: "Agahnim's Tower", slug: "castle_tower", lo: "AP-123", hi: "AP-130", entry: "CastleTowerEntry", leave: "none", open_ids: "AP-123 AP-124 AP-125 AP-126 AP-127 AP-128 AP-129 AP-130", closed: "No closed probe. If the pin report and the disk have no indoor pin, set waiting_on AP-123 and do not boot. Boss file may be created at snes/alttp/dungeon/bosses/agahnim.py.", route: "snes/alttp/docs/routes/castle_tower_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/castle_tower.yaml", boss: "snes/alttp/dungeon/bosses/agahnim.py" },
    #{ name: "Palace of Darkness", slug: "pod", lo: "AP-131", hi: "AP-138", entry: "PalaceOfDarknessEntry", leave: "none", open_ids: "AP-131 AP-132 AP-133 AP-134 AP-135 AP-136 AP-137 AP-138", closed: "PalaceOfDarknessEntry is already indoors. Work interior steps from that pin. AP-097 is not done. Do not walk the pyramid. If AP-134 needs a new hammer primitive in a shared file, note it in the route doc and continue. Boss file may be created at snes/alttp/dungeon/bosses/helmasaur.py.", route: "snes/alttp/docs/routes/pod_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/pod.yaml", boss: "snes/alttp/dungeon/bosses/helmasaur.py" },
    #{ name: "Swamp Palace", slug: "swamp", lo: "AP-139", hi: "AP-146", entry: "SwampPalaceEntry", leave: "AP-139", open_ids: "AP-140 AP-141 AP-142 AP-143 AP-144 AP-145 AP-146", closed: "Do not retry room_engine north_to_0x38 from (4344,1336) layer 1. Start at AP-140. Whatif_SwampPalaceEntry_flippers has flippers $F356=1 and is stuck in room 0x28. If AP-142 needs a new hookshot primitive in a shared file, note it and continue. Boss file may be created at snes/alttp/dungeon/bosses/arrghus.py.", route: "snes/alttp/docs/routes/swamp_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/swamp.yaml", boss: "snes/alttp/dungeon/bosses/arrghus.py" },
    #{ name: "Skull Woods", slug: "skull", lo: "AP-147", hi: "AP-154", entry: "SkullWoodsEntry", leave: "none", open_ids: "AP-147 AP-148 AP-149 AP-150 AP-151 AP-152 AP-153 AP-154", closed: "No SkullWoodsEntry pin was on disk at wave start. If the pin report and the disk still have none, set waiting_on AP-147 and do not boot. Boss file may be created at snes/alttp/dungeon/bosses/mothula.py.", route: "snes/alttp/docs/routes/skull_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/skull.yaml", boss: "snes/alttp/dungeon/bosses/mothula.py" },
    #{ name: "Thieves Town", slug: "thieves", lo: "AP-155", hi: "AP-162", entry: "ThievesTownEntry", leave: "AP-155", open_ids: "AP-156 AP-157 AP-158 AP-159 AP-160 AP-161 AP-162", closed: "Do not retry the lift that cleared the 0x71 tiles and then stuck RIGHT at (6088,6903) in room 0xDB. Start at AP-156 from ThievesTownEntry or Whatif_ThievesTownEntry_lift. Boss file may be created at snes/alttp/dungeon/bosses/blind.py.", route: "snes/alttp/docs/routes/thieves_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/thieves.yaml", boss: "snes/alttp/dungeon/bosses/blind.py" },
    #{ name: "Ice Palace", slug: "ice", lo: "AP-163", hi: "AP-170", entry: "IcePalaceEntry", leave: "AP-165 AP-168", open_ids: "AP-166 AP-169 AP-170", closed: "Do not press A on the room 0x1F big-key chest. AP-168 is claimed by kholdstare-fight. Do not edit kholdstare.py and do not send another Kholdstare stand. Falling ice kills every stand that was tried. AP-169 Needs AP-162 and AP-093. If those are not [x], do not walk the approach. AP-166 is the interior step.", route: "snes/alttp/docs/routes/ice_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/ice.yaml", boss: "snes/alttp/dungeon/bosses/kholdstare.py" },
    #{ name: "Misery Mire", slug: "mire", lo: "AP-171", hi: "AP-178", entry: "MiseryMireEntry", leave: "none", open_ids: "AP-171 AP-172 AP-173 AP-174 AP-175 AP-176 AP-177 AP-178", closed: "No MiseryMireEntry pin was on disk at wave start. If the pin report and the disk still have none, set waiting_on AP-171 and do not boot. Boss file may be created at snes/alttp/dungeon/bosses/vitreous.py.", route: "snes/alttp/docs/routes/mire_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/mire.yaml", boss: "snes/alttp/dungeon/bosses/vitreous.py" },
    #{ name: "Turtle Rock", slug: "turtle_rock", lo: "AP-179", hi: "AP-186", entry: "TurtleRockEntry", leave: "none", open_ids: "AP-179 AP-180 AP-181 AP-182 AP-183 AP-184 AP-185 AP-186", closed: "No TurtleRockEntry pin was on disk at wave start. If the pin report and the disk still have none, set waiting_on AP-179 and do not boot. Boss file may be created at snes/alttp/dungeon/bosses/trinexx.py.", route: "snes/alttp/docs/routes/turtle_rock_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/turtle_rock.yaml", boss: "snes/alttp/dungeon/bosses/trinexx.py" },
    #{ name: "Ganons Tower", slug: "gt", lo: "AP-187", hi: "AP-194", entry: "GanonsTowerEntry", leave: "none", open_ids: "AP-187 AP-188 AP-189 AP-190 AP-191 AP-192 AP-193 AP-194", closed: "No GanonsTowerEntry pin was on disk at wave start. If the pin report and the disk still have none, set waiting_on AP-187 and do not boot. Read your own headings for the boss module name. Do not edit another boss file.", route: "snes/alttp/docs/routes/gt_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/gt.yaml", boss: "snes/alttp/dungeon/bosses/ganon.py" },
];
let jobs = [];
for spec in dungeons {
    jobs.push(#{
        prompt: mgr_prompt(spec.name, spec.slug, spec.lo, spec.hi, spec.entry, spec.leave, spec.open_ids, spec.closed, spec.route, spec.yaml, spec.boss, pin_note),
        label: "mgr-" + spec.lo,
        capability_mode: "all",
        output_schema: lane_schema,
    });
}
let wave = parallel(jobs);
let wave_done = 0;
if wave != () {
    for slot in wave {
        if slot != () && slot.success {
            wave_done += 1;
        }
    }
}
log("dungeon managers returned " + wave_done.to_string());

phase("Report");
let report_prompt = "";
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and one line per dungeon from AP-099 through AP-194 saying which ids are still open. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Dungeon wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
