#!/bin/bash
# usage: bench.sh <label> <server-binary> [env assignments...]
# Starts the server on :8081 with -V, registers the nyx voice, runs the same
# seeded request twice, prints the timing block of the 2nd run + VRAM residency.
set -u
LABEL=$1; BIN=$2; shift 2
S=${BENCH_OUT:-/tmp/qwen3-tts-bench}; mkdir -p "$S"
LOG=$S/$LABEL.log
MODEL=/home/sreed/tools/qwen3-tts.cpp/models/qwen3-tts-0.6b-f16.gguf
TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you, the whole thing fell apart."

pkill -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server' 2>/dev/null; sleep 1
env "$@" "$BIN" -m "$MODEL" -p 8081 -V --seed 42 >"$LOG" 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -sf localhost:8081/v1/audio/voices >/dev/null 2>&1 && break; sleep 0.5; done
VID=$(curl -s -X POST localhost:8081/v1/audio/voices -F name=nyx -F audio_sample=@/home/sreed/tools/reference_voices/nyx_reference.wav | jq -r .id)
for n in 1 2; do
  curl -s -X POST localhost:8081/v1/audio/speech -H 'Content-Type: application/json' \
    -d "{\"input\": $(jq -Rn --arg t "$TEXT" '$t'), \"voice\": \"$VID\", \"seed\": 42}" \
    --output "$S/$LABEL.$n.wav"
done
echo "== $LABEL (pid $SPID) =="
# the real server process may be a child of `toolbox run`; find whoever holds the drm fds
for p in $(pgrep -f 'qwen3-tts-server'); do
  r=$(cat /proc/$p/fdinfo/* 2>/dev/null | awk '/drm-resident-vram/{v+=$2} /drm-resident-gtt/{g+=$2} END{if(v||g) printf "VRAM %.2f GB  GTT %.2f GB", v/1e6, g/1e6}')
  [ -n "$r" ] && echo "residency(pid $p): $r"
done
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
# timing block of the last request
awk '/Talker forward_step/{n++} n==2' "$LOG" | sed -n '1,30p'
grep -E "Code generation|Vocoder decode|Total:|RTF" "$LOG" | tail -4
ls -l "$S/$LABEL".*.wav | awk '{print $5, $9}'
