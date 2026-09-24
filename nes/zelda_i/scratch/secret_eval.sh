#!/usr/bin/env bash
# secret_eval.sh <outdir>: every gather bomb/burn stop x 6 RNG offsets from its predecessor pin. Scratch.
out=$1; mkdir -p "$out"
while read -r pin target; do
  for n in 0 9 18 27 36 45; do
    QT_QPA_PLATFORM=offscreen setsid nohup uv run python nes/zelda_i/scripts/stage_replay.py \
      "$pin" "$target" --assist --idle $n > "$out/$(echo "$target" | tr ':.' '__')_$n.log" 2>&1 &
  done
  wait
done <<'PINS'
GatherChain_walk_7c zelda_i.overworld.gather_segments:make_heart_l8_controller
GatherChain_walk_2c zelda_i.overworld.gather_segments:make_heart_m3_controller
GatherChain_exit_2c zelda_i.scratch.secret_walks:rupees_2d
GatherChain_walk_28 zelda_i.scratch.secret_walks:rupees_28
GatherChain_walk_48 zelda_i.overworld.gather_segments:make_burn_48_controller
GatherChain_exit_48 zelda_i.overworld.gather_segments:make_burn_47_controller
GatherChain_exit_47 zelda_i.scratch.secret_walks:rupees_5b
GatherChain_exit_5b zelda_i.scratch.secret_walks:rupees_6b
GatherChain_exit_6b zelda_i.scratch.secret_walks:rupees_56
GatherChain_exit_ring zelda_i.scratch.secret_walks:rupees_62
GatherChain_exit_62 zelda_i.scratch.secret_walks:potion_from_62
PINS
for f in "$out"/*.log; do
  printf '%s ' "$(basename "$f" .log)"; grep -oE "frames=[0-9]+/[0-9]+ success=\w+" "$f" || echo MISSING
done | sort
