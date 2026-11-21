let meta = #{
    name: "alttp-full-clear",
    description: "Order every z3 item from Sanctuary through the Triforce, then measure the Eastern hall",
    when_to_use: "Plan vanilla 100% autoplay to the credits. Do not replay a blocked probe. The frontier is Play_eog9_7.",
    phases: [
        #{ title: "Census", detail: "item nodes from z3, plus the hall grid" },
        #{ title: "Order", detail: "one route from Sanctuary to the credits" },
        #{ title: "Check", detail: "counts, Triforce, frontier" },
    ],
};

let census_schema = #{
    "type": "object",
    "required": ["out_file", "items", "note"],
    "properties": #{
        "out_file": #{ "type": "string" },
        "items": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

let hall_schema = #{
    "type": "object",
    "required": ["mapped", "path_len", "ran", "room", "note"],
    "properties": #{
        "mapped": #{ "type": "boolean" },
        "path_len": #{ "type": "integer" },
        "ran": #{ "type": "boolean" },
        "room": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
};

let order_schema = #{
    "type": "object",
    "required": ["items", "ends_at", "frontier", "note"],
    "properties": #{
        "items": #{ "type": "integer" },
        "ends_at": #{ "type": "string" },
        "frontier": #{ "type": "string" },
        "note": #{ "type": "string" },
    },
};

let check_schema = #{
    "type": "object",
    "required": ["ok", "missing", "note"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "missing": #{ "type": "integer" },
        "note": #{ "type": "string" },
    },
};

fn census_prompt(label, roots, out_file) {
    let p = "";
    p += "Repo: /home/v/01_projects/11_games/retro. You are one census lane, ";
    p += label;
    p += ". Read every JSON file under these roots, relative to snes/alttp/z3-json-data/: ";
    p += roots;
    p += ".\n";
    p += "Use read_file and grep. Do not answer from memory. Prefer snes/alttp/refs/z3-json-data when that file exists, else snes/alttp/z3-json-data. Do not commit refs. Do not download.\n";
    p += "Write only ";
    p += out_file;
    p += ". One line per node that has nodeItem, or whose name is Ganon. Columns: file, node name, nodeItem or none, requires copied from the JSON or none. Then a final line COUNT N. N is the number of nodeItem lines, not the Ganon drop.\n";
    p += "Quote requires. Do not invent a requirement, a room id, or an item. Skip nodes with no nodeItem unless the name is Ganon.\n";
    p += "If a requires list mentions Crystal, Pendant, BigKey, or RescueZelda, copy that list in full.\n";
    p += "Do not edit PORT_PLAN.md, docs/STATUS.md, ITEM_ROUTE.md, room_engine.py, primitives.py, maps, or another lane's file. Do not boot a ROM. Do not run bd. Do not git commit.\n";
    p += "Return out_file, items (the COUNT), and note (one sentence: anything whose requires you could not quote).\n";
    return p;
}

let hall_prompt = "";
hall_prompt += "Repo: /home/v/01_projects/11_games/retro. You are the Eastern hall lane. The continuous predecessor is Play_eog9_7, room 0xB9, indoors, module 0x07, (4856, 6104), sword 1, lamp 1, HP 28/32, follower 0, keys 0. Replay recordings/eog9_lobby_20261023/replay.json is 602 frames, desync 0, loads 0, writes 0. AP-105 stays open. Do not tick it.\n";
hall_prompt += "The north door in maps/room_b9.json is UP into room 0xA9 and has no approach point. The west door has no floor path. The east door is type 0x1C and keys are 0. Do not press A on the rupee chest. Do not write $F3CC, sword, keys, HP, rupees, or the big-key bit.\n";
hall_prompt += "Do not edit opening_route/room_engine.py or primitives.py. Do not add a sibling module. Geometry goes in maps/room_b9.json only if find_path returns cells. Do not invent an approach xy.\n";
hall_prompt += "From the repo root, with SDL_VIDEODRIVER=dummy and PYTHONPATH=/home/v/01_projects/11_games/retro:/home/v/01_projects/11_games/retro/snes:/home/v/01_projects/11_games/retro/nes, run uv run python -m alttp.scripts.pin_probe Play_eog9_7 --doors --tiles --fixture. Read pin_probe.write_fixture before you depend on the output path. Do not overwrite tests/fixtures/room_61.json.\n";
hall_prompt += "Dump both attribute layers with the existing read_room_tiles while that pin is open. A /tmp script is allowed. Do not commit it. Call alttp.dungeon.nav.find_path from the south landing tile to the north door tile on the layer Link is actually standing on. If that path is empty, try the other layer once. If both are empty, mapped stays false, path_len is 0, ran stays false, and you do not run room_engine.\n";
hall_prompt += "If a path exists, write those measured cells into maps/room_b9.json as points and set the north door approachXy to the last floor cell before the door. Then run room_engine once: north, state Play_eog9_7, cwd snes, SDL_VIDEODRIVER=dummy. Quote the leave glance. One run. If it fails, stop. Do not retry the west door or the east lock.\n";
hall_prompt += "Do not edit PORT_PLAN.md or docs/STATUS.md. Do not run bd. Do not git commit. Do not overwrite recordings/clean_escape_20261002 or recordings/eog9_lobby_20261023.\n";
hall_prompt += "Return mapped, path_len, ran, room (the leave room or 0xB9), and note (layer, path, glance, or why the path was empty).\n";

phase("Census");
let jobs = [
    #{
        prompt: census_prompt(
            "lightworld-main",
            "regions/lightworld/main.json",
            "snes/alttp/docs/routes/full_clear/lightworld_main.md"
        ),
        label: "census-lw-main",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "lightworld-mountain",
            "regions/lightworld/deathmountain/",
            "snes/alttp/docs/routes/full_clear/lightworld_mountain.md"
        ),
        label: "census-lw-mountain",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "lightworld-field",
            "regions/lightworld/northeast/ regions/lightworld/northwest/ regions/lightworld/south/",
            "snes/alttp/docs/routes/full_clear/lightworld_field.md"
        ),
        label: "census-lw-field",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "darkworld-mountain",
            "regions/darkworld/deathmountain/",
            "snes/alttp/docs/routes/full_clear/darkworld_mountain.md"
        ),
        label: "census-dw-mountain",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "darkworld-field",
            "regions/darkworld/east/ regions/darkworld/south/ regions/darkworld/west/ regions/darkworld/mire/ regions/darkworld/northeast/ regions/darkworld/northwest/",
            "snes/alttp/docs/routes/full_clear/darkworld_field.md"
        ),
        label: "census-dw-field",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "dungeons-early",
            "regions/dungeons/escape/ regions/dungeons/eastern/ regions/dungeons/desert/ regions/dungeons/hera/",
            "snes/alttp/docs/routes/full_clear/dungeons_early.md"
        ),
        label: "census-early",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "dungeons-mid",
            "regions/dungeons/castletower/ regions/dungeons/pod/ regions/dungeons/swamp/ regions/dungeons/skwoods/",
            "snes/alttp/docs/routes/full_clear/dungeons_mid.md"
        ),
        label: "census-mid",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: census_prompt(
            "dungeons-late",
            "regions/dungeons/thieves/ regions/dungeons/ice/ regions/dungeons/mire/ regions/dungeons/trock/ regions/dungeons/gt/",
            "snes/alttp/docs/routes/full_clear/dungeons_late.md"
        ),
        label: "census-late",
        capability_mode: "read-write",
        output_schema: census_schema,
    },
    #{
        prompt: hall_prompt,
        label: "hall-0xb9",
        capability_mode: "all",
        output_schema: hall_schema,
    },
];
let census = parallel(jobs);

let item_sum = 0;
let census_note = "";
let hall_note = "hall lane returned nothing";
let hall_mapped = false;
let hall_ran = false;
let hall_room = "0xB9";
for r in census {
    if r == () || r.success != true || type_of(r.output) != "map" {
        census_note += "one lane failed. ";
        continue;
    }
    if r.output.items != () {
        item_sum += r.output.items;
        census_note += r.output.out_file;
        census_note += "=";
        census_note += r.output.items.to_string();
        census_note += ". ";
    }
    if r.output.path_len != () {
        hall_note = if r.output.note != () { r.output.note } else { "hall note empty" };
        hall_mapped = r.output.mapped == true;
        hall_ran = r.output.ran == true;
        if r.output.room != () { hall_room = r.output.room; }
    }
}
log("census items " + item_sum.to_string());

phase("Order");
let order_prompt = "";
order_prompt += "Repo: /home/v/01_projects/11_games/retro. You are the route manager. Read every file in snes/alttp/docs/routes/full_clear/ except ORDER.md, plus snes/alttp/docs/routes/ITEM_ROUTE.md and helpers.json in the z3 pin (refs if present, else snes/alttp/z3-json-data/helpers.json). Use read_file and grep.\n";
order_prompt += "Write only snes/alttp/docs/routes/full_clear/ORDER.md. Do not edit PORT_PLAN.md, docs/STATUS.md, ITEM_ROUTE.md, or the census files.\n";
order_prompt += "The Clean tape recordings/clean_escape_20261002 ends in room 0x12. That is leg 0. The continuous predecessor now is Play_eog9_7 in room ";
order_prompt += hall_room;
order_prompt += ".\n";
order_prompt += "Hall measurement: mapped=";
order_prompt += if hall_mapped { "true" } else { "false" };
order_prompt += " ran=";
order_prompt += if hall_ran { "true" } else { "false" };
order_prompt += " note: ";
order_prompt += hall_note;
order_prompt += "\n";
order_prompt += "List every census nodeItem once. Order them so a requires token is collected before it is used. Expand helpers from helpers.json. If a token has no expansion and is not itself a nodeItem, name it in the note and keep the line.\n";
order_prompt += "Ganon's nodeItem is Triforce. The node in regions/darkworld/northeast/caves.json has no requires list. Find the crystal or prize gate on a connection or a dungeon prize node and quote that JSON. If the pin does not encode seven crystals on the pyramid, write that, and do not invent the gate.\n";
order_prompt += "The last leg is the credits. AP-197 asks for a Triforce-room stop plus the credits module. Neither is implemented. Do not claim a credits tape.\n";
order_prompt += "The frontier is the first leg that is not already true on the Clean tape or on Play_eog9_7. Do not mark a blocked big-key walk, a 0/12 boss, or a claimed fight as runnable. AP-113 and AP-168 stay claimed. Do not boot from this file.\n";
order_prompt += "Census lanes reported about ";
order_prompt += item_sum.to_string();
order_prompt += " items. ";
order_prompt += census_note;
order_prompt += "\nReturn items (lines in ORDER that name a nodeItem), ends_at (Triforce then credits, or the gap), frontier (one leg), and note.\n";
let order = agent(order_prompt, #{
    label: "mgr-order",
    capability_mode: "read-write",
    output_schema: order_schema,
});
let order_items = 0;
let ends_at = "missing";
let frontier = "missing";
let order_note = "order lane returned nothing";
if order != () && order.success == true && type_of(order.output) == "map" {
    if order.output.items != () { order_items = order.output.items; }
    if order.output.ends_at != () { ends_at = order.output.ends_at; }
    if order.output.frontier != () { frontier = order.output.frontier; }
    if order.output.note != () { order_note = order.output.note; }
}
log("order items " + order_items.to_string());

phase("Check");
let check_prompt = "";
check_prompt += "Repo: /home/v/01_projects/11_games/retro. Check snes/alttp/docs/routes/full_clear/ORDER.md against the other files in that directory. Use read_file and grep. Do not boot a ROM.\n";
check_prompt += "ok is true only when all of these hold: every census COUNT is present as that many nodeItem lines in ORDER.md, Triforce appears once, the last leg says credits and does not claim a tape exists, and the frontier is one uncollected leg. missing is the number of census items absent from ORDER.md.\n";
check_prompt += "Manager said items ";
check_prompt += order_items.to_string();
check_prompt += ", ends_at ";
check_prompt += ends_at;
check_prompt += ", frontier ";
check_prompt += frontier;
check_prompt += ", note ";
check_prompt += order_note;
check_prompt += ".\n";
check_prompt += "If ok is false, edit only ORDER.md to add the missing nodeItem lines and the quoted gap. Do not invent requires. Do not edit PORT_PLAN.md or docs/STATUS.md.\n";
check_prompt += "Return ok, missing, and note.\n";
let check = agent(check_prompt, #{
    label: "check-order",
    capability_mode: "read-write",
    output_schema: check_schema,
});
let ok = false;
let missing = -1;
let check_note = "check lane returned nothing";
if check != () && check.success == true && type_of(check.output) == "map" {
    ok = check.output.ok == true;
    if check.output.missing != () { missing = check.output.missing; }
    if check.output.note != () { check_note = check.output.note; }
}
let summary = "full clear order ok=";
summary += if ok { "true" } else { "false" };
summary += " missing=";
summary += missing.to_string();
summary += " frontier=";
summary += frontier;
summary += " hall=";
summary += hall_room;
summary += " ";
summary += check_note;
complete(#{
    ok: ok,
    missing: missing,
    items: order_items,
    frontier: frontier,
    ends_at: ends_at,
    hall_room: hall_room,
    hall_note: hall_note,
    summary: summary,
});
