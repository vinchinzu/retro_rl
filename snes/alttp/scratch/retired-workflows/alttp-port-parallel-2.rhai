let meta = #{
    name: "alttp-port-parallel-2",
    description: "Finish the Ice big-key claim, and retry four item gaps with a new inventory write",
    when_to_use: "After alttp-port-parallel. Ice AP-165 is claimed. Do not repeat the failed chest walks.",
    phases: [
        #{ title: "Lanes", detail: "Ice plus four item gaps" },
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
    p += "Repo: /home/v/01_projects/11_games/retro_rl. Tracker ticks only through python3 /home/v/.grok/alttp_port_claim.py claim|done|block|note. Do not edit PORT_PLAN.md in an editor. Do not run bd. Do not git add, commit, or checkout. Do not edit docs/STATUS.md.\n";
    p += "A pin is isolated development. It is not natural, not continuous, and not a STATUS claim. Never write $F3CC. Do not overwrite recordings/verified_tip_run.json, EasternPalaceEntry, DesertPalaceEntry, HeraEntry, IcePalaceEntry, HyruleCastleGrounds, Scan_* pins, or Play_* pins.\n";
    p += "whatif_pin.py STATE --set FIELD=VALUE --out Whatif_<name> copies a pin. Read snes/alttp/ram.py for field names before you choose one. pin_probe.py only reads. cwd for ROM commands is snes/. Set SDL_VIDEODRIVER=dummy. One emulator. Do not kill other Python processes.\n";
    p += "Read your own headings in snes/alttp/docs/PORT_PLAN.md and snes/alttp/AGENTS.md. Use read_file, grep, and the shell. Do not answer from memory. Re-read a file immediately before you edit it. Soft max about 1000 lines.\n";
    p += "Mark a task done only when its own Done check passes. If the Done line requires the item to rise in play, a poke does not pass it. Two honest tries of a new method, then block. Do not repeat a walk that the existing block text already records.\n";
    p += "Return completed (AP ids you marked [x]), waiting_on (one id or empty), blocked (one id or empty).\n";
    return p;
}

phase("Lanes");
let jobs = [];

let ice = common();
ice += "You are the Ice Palace manager. Claim name is exactly Ice Palace, because AP-165 is already [~] Ice Palace.\n";
ice += "Do AP-165, then AP-166 only if its Needs are all [x]. AP-163 and AP-164 are done. Do not redo them.\n";
ice += "Learned fact: Ice LEFT 72 coasts 18 px after the 1.5 px/frame cap. Rooms 0x0E and 0x1E have no chest. East and west doors 0x18 and the south door 0x00 stayed shut. Big key $F366=0x0040 on Whatif_IcePalaceEntry_parallel is a poke and does not close a task whose Done line says the key rose in play.\n";
ice += "You may edit snes/alttp/walk_physics.py only to add that measured coast. You may edit snes/alttp/docs/routes/ice_ROUTE.md, snes/alttp/dungeon/route_data/ice.yaml, and maps for rooms your tasks name. Do not edit nav.py, primitives.py, or another dungeon's files.\n";
ice += "Find the big-key room from the ice route yaml or z3 json. Do not invent a chest in 0x0E or 0x1E.\n";
jobs.push(#{
    prompt: ice,
    label: "mgr-ice",
    capability_mode: "all",
    output_schema: lane_schema,
});

let desert = common();
desert += "You are the Desert glove lane. Do not repeat the room 0x73 shutter, the pot lifts, or the center wall cols 27-36. Those tries left $F354 at 0.\n";
desert += "Copy Whatif_DesertPalaceEntry_parallel to Whatif_DesertPalaceEntry_glove with gloves=1. Read the byte back with pin_probe.\n";
desert += "The block text says the rock back entrance was not walked because the glove was 0. Walk that entrance once from the new pin. If you reach the Lanmolas arena, claim AP-111 and do the probe, then AP-112 only when the probe is real. Boss step 6 needs at least 10 of 12 RNG offsets. If the arena stays closed, note AP-110, AP-111, and AP-112 and do not mark them done.\n";
desert += "You may edit snes/alttp/docs/routes/desert_ROUTE.md and desert recordings under snes/alttp/recordings/probe_AP-111/. Do not edit shared modules.\n";
jobs.push(#{
    prompt: desert,
    label: "mgr-desert-glove",
    capability_mode: "all",
    output_schema: lane_schema,
});

let hera = common();
hera += "You are the Hera lane. Leave AP-118 blocked. Do not rewalk the Moon Pearl chest, the peg floor, or the pit drop already in that block text. Pearl $F357 stayed 0.\n";
hera += "Copy Whatif_HeraEntry_parallel to Whatif_HeraEntry_pearl with pearl=1. Then do open tasks AP-119 and AP-120 only. Read their Done lines. If you cannot reach Moldorm from this pin by a door the block text does not already call a failure, leave AP-119 open, note the pin, and stop. Do not block AP-119 for an unwalked path.\n";
hera += "You may edit snes/alttp/docs/routes/hera_ROUTE.md and recordings under snes/alttp/recordings/probe_AP-119/. Do not edit shared modules.\n";
jobs.push(#{
    prompt: hera,
    label: "mgr-hera-pearl",
    capability_mode: "all",
    output_schema: lane_schema,
});

let swamp = common();
swamp += "You are the Swamp lane. AP-139 is blocked because north_to_0x38 stuck at (4344,1376) in room 0x28 with flippers 0. That hold-UP result is closed. Do not repeat it as the only try.\n";
swamp += "Copy SwampPalaceEntry to Whatif_SwampPalaceEntry_flippers with flippers=1. Prove the byte with pin_probe. One new hop toward the 0x38 stairs. If room 0x38 loads, claim AP-139 and mark it done with the pin name and the flippers write. If it stays in 0x28, block AP-139 again with the new pose and flippers value.\n";
swamp += "You may edit snes/alttp/docs/routes/swamp_ROUTE.md and maps for rooms you measure. Do not edit shared modules.\n";
jobs.push(#{
    prompt: swamp,
    label: "mgr-swamp-flippers",
    capability_mode: "all",
    output_schema: lane_schema,
});

let thieves = common();
thieves += "You are the Thieves lane. AP-155 is blocked: east_to_0xDC stayed in 0xDB, and a bomb left the 0x71 tiles unchanged. Ability $F379 was 0x12, which lacks lift bit 0x40. Do not repeat that bomb.\n";
thieves += "Read the ability byte on ThievesTownEntry, then copy it to Whatif_ThievesTownEntry_lift with ability set to that byte OR 0x40. Do not clear the other bits. Prove the byte with pin_probe.\n";
thieves += "One new east hop. If room 0xDC loads, claim AP-155 and mark it done with the pin name and the ability write. If it stays in 0xDB, block AP-155 again with the new pose and the ability byte.\n";
thieves += "You may edit snes/alttp/docs/routes/thieves_ROUTE.md and maps for rooms you measure. Do not edit shared modules.\n";
jobs.push(#{
    prompt: thieves,
    label: "mgr-thieves-lift",
    capability_mode: "all",
    output_schema: lane_schema,
});

let eastern = common();
eastern += "You are the Eastern bow lane. AP-102 is blocked. The closed fact is room 0xA9 pose (4856,5592), bow 0, arrows 0, layer-1 chest (30,26) with no layer-0 path. Do not rewalk that chest.\n";
eastern += "Copy Whatif_EasternPalaceEntry_parallel to Whatif_EasternPalaceEntry_bow with bow=1 and arrows greater than 0. Read both bytes back.\n";
eastern += "Read the AP-102 Done line. If it requires the chest prize to rise in play, leave AP-102 blocked and note the pin. If it requires a live shoot_arrow check, add that primitive only in the file the task names, test it from room 0xC9 on the new pin, and mark AP-102 done only when that check passes. Evidence must say isolated poke.\n";
eastern += "AP-103 and AP-104 are open. Try them only after a path to the Armos room that is not the failed 0xA9 chest path. If no such path is in the route yaml, leave them open and note that. Do not block them for an unwalked path.\n";
eastern += "You may edit the shoot_arrow file the task names, snes/alttp/docs/routes/eastern_ROUTE.md, and recordings under snes/alttp/recordings/probe_AP-102/. Do not edit walk_physics.py.\n";
jobs.push(#{
    prompt: eastern,
    label: "mgr-eastern-bow",
    capability_mode: "all",
    output_schema: lane_schema,
});

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
report_prompt += "Count heading ticks in /home/v/01_projects/11_games/retro_rl/snes/alttp/docs/PORT_PLAN.md with Python. A heading starts with '- [' and contains **AP-. open_count is a space box. done_count is x. blocked_count is !. summary: those three numbers, the blocked ids, the lowest open id, and the box mark for AP-102, AP-139, AP-155, and AP-165. Do not edit files.\n";
let report = agent(report_prompt, #{
    label: "mgr-report",
    capability_mode: "execute",
    output_schema: report_schema,
});
if report != () && report.success && type_of(report.output) == "map" && report.output.summary != () {
    complete(report.output);
}
complete(#{
    summary: "Follow-up finished but the report manager did not return counts.",
    open_count: -1,
    done_count: -1,
    blocked_count: -1,
});
