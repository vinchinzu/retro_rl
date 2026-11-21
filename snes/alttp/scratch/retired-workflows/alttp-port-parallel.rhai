let meta = #{
    name: "alttp-port-parallel",
    description: "Generate dungeon-start save states with inventory writes, unblock what that opens, then run the twelve dungeon managers together",
    when_to_use: "After ability-play left the big keys blocked. Owner allowed entry pins and inventory RAM writes so dungeons can run in parallel.",
    phases: [
        #{ title: "Unblock", detail: "entry pins plus any ticket a pin can open" },
        #{ title: "Dungeons", detail: "twelve managers from their start pins" },
        #{ title: "Report", detail: "tick counts" },
    ],
};

let pin_schema = #{
    "type": "object",
    "required": ["pins_ready", "still_blocked", "opened", "note"],
    "properties": #{
        "pins_ready": #{ "type": "array", "items": #{ "type": "string" } },
        "still_blocked": #{ "type": "array", "items": #{ "type": "string" } },
        "opened": #{ "type": "array", "items": #{ "type": "string" } },
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

fn mgr_prompt(name, slug, lo, hi, entry, skip, route, yaml, pin_note) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro_rl. You are one dungeon manager. Agent name ";
    p += name;
    p += ". Dungeon slug ";
    p += slug;
    p += " only. Tasks ";
    p += lo;
    p += " through ";
    p += hi;
    p += " in order.\n";
    p += "Tracker: snes/alttp/docs/PORT_PLAN.md. Ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit that file in an editor. Do not run bd. Do not git add, commit, or checkout. Do not revert unrelated dirty files.\n";
    p += "Read snes/alttp/AGENTS.md, the eight-step dungeon template in snes/alttp/docs/routes/DUNGEON_TEMPLATE.md, and only your own headings in PORT_PLAN.md. Do not read other dungeons.\n";
    p += "Owner ruling 2026-09-30: a generated start pin with disclosed inventory RAM writes is enough to work this dungeon in parallel. The pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json.\n";
    p += "Start pin: prefer Whatif_";
    p += entry;
    p += "_parallel when that file exists, else ";
    p += entry;
    p += ". The unblock report JSON is below. If neither pin exists, set waiting_on to your first open id, leave blocked empty, and do not boot a ROM.\n";
    p += "Unblock report JSON: ";
    p += pin_note;
    p += "\n";
    if skip != "" {
        p += "Leave ";
        p += skip;
        p += " blocked. Do not claim it, do not mark it done, and do not repeat its probe. Steps after it omit a Needs line; you may still run those later steps from the start pin. Evidence must name the pin and say isolated poke.\n";
    }
    p += "Before each task, re-read its heading. Skip [x]. If it is [~] or [!] for someone else, and it is not the skip id above, stop the lane. If a Needs id is not [x], stop and set waiting_on. Do not mark a waiting task blocked.\n";
    p += "A step is done only when its own Done check passes. Two honest tries, then block that step. Do not weaken the Done check. If the Done line requires the item to rise in play, a poke does not pass it: leave the task and note the pin.\n";
    p += "Boss step 6 needs at least 10 of 12 RNG offsets. Hint lines are not authority. z3-json-data and a live scan win. Prefer snes/alttp/refs/z3-json-data when present, else snes/alttp/z3-json-data. Do not commit refs and do not download.\n";
    p += "You may edit only these paths: ";
    p += route;
    p += ", ";
    p += yaml;
    p += ", maps/room_*.json for rooms your own tasks name, scratch/probe_ap files whose id is inside your range, and recordings under snes/alttp/recordings/probe_";
    p += lo;
    p += "/. Re-read a file immediately before you edit it. Yaml edits are hint fixes only.\n";
    p += "Do not edit make_entry_pin.py, whatif_pin.py, hops.py, primitives.py, ram.py, nav.py, tilemap.py, paths.py, AGENTS.md, docs/STATUS.md, docs/ASSIST_CONTRACT.md, or another dungeon's files. If a task needs one of those shared files, set waiting_on to that task id and write the needed change in your route doc.\n";
    p += "Commands that boot the ROM use cwd snes/ and SDL_VIDEODRIVER=dummy. One emulator in your process. Do not kill other Python processes. Pytest uses QT_QPA_PLATFORM=offscreen. Soft max about 1000 lines. Geometry stays in maps json. Do not copy approach xy into Python.\n";
    p += "Use read_file, grep, and the shell. Do not answer from memory.\n";
    p += "Return completed (AP ids you marked [x]), waiting_on (one id or empty), blocked (one id or empty).\n";
    return p;
}

phase("Unblock");
let unblock_prompt = "";
unblock_prompt += "Repo: /home/v/01_projects/11_games/retro_rl. You are mgr-unblock, the only agent who may create dungeon-start pins. You do not implement dungeon steps AP-099 through AP-194.\n";
unblock_prompt += "Tracker ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md.\n";
unblock_prompt += "Owner ruling 2026-09-30: generate a save state at the start of each dungeon, and use disclosed RAM writes for inventory, keys, magic, and ability bits the later steps need, so twelve dungeon managers can run in parallel. Pins are development only. Provenance must say poke. Never write $F3CC. Do not call a pin natural or continuous. Do not overwrite recordings/verified_tip_run.json.\n";
unblock_prompt += "Do not retry these measurements. They are closed facts: Desert Whatif_DesertPalaceEntry_abilityFC and A at sprite 0x3B in room 0x73 (keys stayed 0); Eastern magic $F36E=128 on the big-key chest; the Hera basement wall on row 52 cols 27-36; any further search for a writer of bit 0x04 of $7EF379. The headerless ROM sets that bit only with STA.l at $09:8658 after ORA.l at $09:8654, and only when Y is $4B and $F355 is not already 1. DesertPalaceEntry already has the boots byte set.\n";
unblock_prompt += "Do not mark AP-101, AP-109, or AP-117 done. A poke is not a key that rose in play. Leave those three blocked. Add one note on each naming the parallel pin later steps may load.\n";
unblock_prompt += "Do not overwrite EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, HyruleCastleGrounds, Scan_* pins, or Play_* pins.\n";
unblock_prompt += "Read these files before you edit or boot: snes/alttp/scripts/make_entry_pin.py, snes/alttp/scripts/whatif_pin.py, snes/alttp/ram.py inventory fields, and snes/alttp/AGENTS.md. cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator at a time. Do not kill other Python processes.\n";
unblock_prompt += "make_entry_pin.py <slug> loads HyruleCastleGrounds, pokes the bring-in kit, triggers the entrance, and saves the *Entry pin with provenance poke. ENTRIES currently has eastern and ice. It refuses $F3CC and refuses to overwrite a pin that lands in a different room. --print-kit prints the kit and does not boot. whatif_pin.py STATE --set FIELD=VALUE --out Whatif_<name> copies a pin and writes named inventory fields. It refuses $F3CC. pin_probe.py only reads.\n";
unblock_prompt += "Find the real .state directory under snes/alttp/custom_integrations/ with a search. Do not guess a second path.\n";
unblock_prompt += "For eastern, desert, hera, and ice: if the *Entry pin exists, copy it with whatif_pin.py to Whatif_<Entry>_parallel. Write the inventory the open later steps need. A disclosed big-key or small-key write is allowed on that Whatif copy so later steps can run. Do not walk the failed torch, cone, or Hera wall. If the big-key word is not a named inventory field, extend whatif_pin.py or make_entry_pin.py with a poke that still refuses $F3CC, still writes provenance poke, and still refuses to clobber the source pin. Re-read the file immediately before the edit. Verify each new pin with pin_probe: room, module, and the bytes you wrote.\n";
unblock_prompt += "For castle_tower, pod, swamp, skull, thieves, mire, turtle_rock, and gt: if no indoor *Entry pin exists, add an EntrySpec only after you have read the entrance id from snes/alttp/dungeon/route_data/<slug>.yaml or the z3 region json. Do not invent an entrance id. Run --print-kit, then one rebuild. If the room guard refuses the overwrite, save under a new *Entry name the tool already allows rather than clobbering a scan pin. If you cannot find an entrance id, skip that slug.\n";
unblock_prompt += "Then read only these headings, with grep, not the whole plan: AP-022, AP-064, AP-067, AP-076, AP-086, AP-093, AP-094, AP-096, and AP-068. If the block note is the measurement already recorded, do not repeat that probe. If a disclosed Whatif pin meets that task's own Done line, mark it done and name the pin and the writes in the evidence. If the Done line requires the change to rise in play, leave the [!] and add a note. If AP-022 becomes [x], claim and do AP-068. If AP-022 stays [!], do not claim AP-068.\n";
unblock_prompt += "Use read_file, grep, and the shell. Do not answer from memory. Soft max about 1000 lines.\n";
unblock_prompt += "Return pins_ready (pin names pin_probe showed indoors), still_blocked (AP ids still [!]), opened (AP ids you marked [x]), and note (one paragraph: pin name per dungeon slug, and which bytes were writes).\n";
let unblock = agent(unblock_prompt, #{
    label: "mgr-unblock",
    capability_mode: "all",
    output_schema: pin_schema,
});
let pin_note = "{}";
if unblock == () || unblock.success != true {
    log("unblock manager returned no success; dungeon managers still check for pins");
} else if type_of(unblock.output) == "map" {
    pin_note = json_encode(unblock.output);
    log("unblock manager returned a pin report");
}

phase("Dungeons");
let dungeons = [
    #{ name: "Eastern Palace", slug: "eastern", lo: "AP-099", hi: "AP-106", entry: "EasternPalaceEntry", skip: "AP-101", route: "snes/alttp/docs/routes/eastern_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/eastern.yaml" },
    #{ name: "Desert Palace", slug: "desert", lo: "AP-107", hi: "AP-114", entry: "DesertPalaceEntry", skip: "AP-109", route: "snes/alttp/docs/routes/desert_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/desert.yaml" },
    #{ name: "Tower of Hera", slug: "hera", lo: "AP-115", hi: "AP-122", entry: "HeraEntry", skip: "AP-117", route: "snes/alttp/docs/routes/hera_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/hera.yaml" },
    #{ name: "Agahnim's Tower", slug: "castle_tower", lo: "AP-123", hi: "AP-130", entry: "CastleTowerEntry", skip: "", route: "snes/alttp/docs/routes/castle_tower_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/castle_tower.yaml" },
    #{ name: "Palace of Darkness", slug: "pod", lo: "AP-131", hi: "AP-138", entry: "PalaceOfDarknessEntry", skip: "", route: "snes/alttp/docs/routes/pod_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/pod.yaml" },
    #{ name: "Swamp Palace", slug: "swamp", lo: "AP-139", hi: "AP-146", entry: "SwampPalaceEntry", skip: "", route: "snes/alttp/docs/routes/swamp_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/swamp.yaml" },
    #{ name: "Skull Woods", slug: "skull", lo: "AP-147", hi: "AP-154", entry: "SkullWoodsEntry", skip: "", route: "snes/alttp/docs/routes/skull_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/skull.yaml" },
    #{ name: "Thieves' Town", slug: "thieves", lo: "AP-155", hi: "AP-162", entry: "ThievesTownEntry", skip: "", route: "snes/alttp/docs/routes/thieves_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/thieves.yaml" },
    #{ name: "Ice Palace", slug: "ice", lo: "AP-163", hi: "AP-170", entry: "IcePalaceEntry", skip: "", route: "snes/alttp/docs/routes/ice_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/ice.yaml" },
    #{ name: "Misery Mire", slug: "mire", lo: "AP-171", hi: "AP-178", entry: "MiseryMireEntry", skip: "", route: "snes/alttp/docs/routes/mire_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/mire.yaml" },
    #{ name: "Turtle Rock", slug: "turtle_rock", lo: "AP-179", hi: "AP-186", entry: "TurtleRockEntry", skip: "", route: "snes/alttp/docs/routes/turtle_rock_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/turtle_rock.yaml" },
    #{ name: "Ganon's Tower", slug: "gt", lo: "AP-187", hi: "AP-194", entry: "GanonsTowerEntry", skip: "", route: "snes/alttp/docs/routes/gt_ROUTE.md", yaml: "snes/alttp/dungeon/route_data/gt.yaml" },
];
let jobs = [];
for spec in dungeons {
    jobs.push(#{
        prompt: mgr_prompt(spec.name, spec.slug, spec.lo, spec.hi, spec.entry, spec.skip, spec.route, spec.yaml, pin_note),
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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and whether AP-101, AP-109, and AP-117 are x, space, or !. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Parallel wave finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
