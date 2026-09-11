#!/bin/bash
# usage: bench-hip.sh <label> <binary> [ENV=VAL ...]   (runs inside navi31-llama)
# Warm-up with a 1-word request (tiny vocoder decode), then the full text; the
# server is killed as soon as the 2nd timing block prints, so the pathologically
# slow HIP vocoder never runs on the long text.
set -u
LABEL=$1; BIN=$2; shift 2
S=${BENCH_OUT:-/tmp/qwen3-tts-bench}; mkdir -p "$S"; LOG=$S/$LABEL.log
MODEL=/home/sreed/tools/qwen3-tts.cpp/models/qwen3-tts-0.6b-f16.gguf
TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you, the whole thing fell apart."
pkill -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server' 2>/dev/null; sleep 1
toolbox run --container navi31-llama env "$@" "$BIN" -m "$MODEL" -p 8081 -V --seed 42 >"$LOG" 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8081/v1/audio/voices >/dev/null 2>&1 && break; sleep 0.5; done
VID=$(curl -s -X POST localhost:8081/v1/audio/voices -F name=nyx -F audio_sample=@/home/sreed/tools/reference_voices/nyx_reference.wav | jq -r .id)
req() { curl -s -m 300 -X POST localhost:8081/v1/audio/speech -H 'Content-Type: application/json' \
    -d "{\"input\": $(jq -Rn --arg t "$1" '$t'), \"voice\": \"$VID\", \"seed\": 42}" --output "$S/$LABEL.$2.wav"; }
req "Hi." 1
req "$TEXT" 2 &
for i in $(seq 1 600); do [ "$(grep -c 'Speech codes generated' "$LOG")" -ge 2 ] && break; sleep 0.2; done
for p in $(pgrep -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server'); do
  cat /proc/$p/fdinfo/* 2>/dev/null | awk -v p=$p '/drm-resident-vram/{v+=$2} /drm-resident-gtt/{g+=$2} END{if(v||g) printf "residency(pid %d): VRAM %.2f GB  GTT %.2f GB\n", p, v/1e6, g/1e6}'
done
pkill -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server'; wait 2>/dev/null
echo "== $LABEL =="
awk '/Talker forward_step/{n++} n==2' "$LOG" | grep -E "Total:|Steps|Compute|Data I/O|Graph alloc|Graph build|Embed|Throughput|Speech codes" | head -16
