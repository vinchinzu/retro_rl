#!/usr/bin/env bash
# ab_stage.sh <outdir> <pin-suffix> <module:factory> [pins...]: 12 offsets x pins, summary. Scratch.
out=$1; suffix=$2; target=$3; shift 3
pins=${@:-BlueRingFull14 BlueRingFull10 BlueRingFull3}
mkdir -p "$out"
tag=$(echo "$target" | tr ':.' '__')
for pin in $pins; do for n in 0 7 14 21 28 35 42 49 56 63 70 77; do
  QT_QPA_PLATFORM=offscreen setsid nohup uv run python nes/zelda_i/scripts/stage_replay.py \
    ${pin}_${suffix} "$target" --assist --idle $n > "$out/${tag}_${pin}_$n.log" 2>&1 &
done; done
wait
python3 - "$out" "$tag" <<'PY'
import glob, re, statistics as st, sys
out, tag = sys.argv[1], sys.argv[2]
fr, dm, ok = [], [], 0
for f in glob.glob(f"{out}/{tag}_*.log"):
    t = open(f).read()
    m = re.search(r"frames=(\d+)/\d+ success=(\w+)", t); d = re.search(r"damage ([\d.]+)h", t)
    if not m or not d: print("BAD", f); continue
    fr.append(int(m.group(1))); dm.append(float(d.group(1))); ok += m.group(2) == "True"
print(f"{tag}: ok {ok}/{len(fr)} frames {st.mean(fr):.0f} (sd {st.stdev(fr):.0f}) hearts {st.mean(dm):.2f} (sd {st.stdev(dm):.2f}) max {max(dm):.1f}")
PY
