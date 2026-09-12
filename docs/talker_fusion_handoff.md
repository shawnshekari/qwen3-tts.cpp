# Talker Fusion — Session Handoff

Start here if you're picking up the fused HIP talker. Written 2026-09-12
after the Phase 3 implementation (see `code_predictor_plan.md` Phase 3
for the kernel design and parity numbers). This doc covers the **ship
decision** and the **next step**, not the kernel internals.

## Current state (what's done, what's live)

- Fused talker kernel: `src/hip/talker_hip.{h,hip}`, built and parity-
  verified (hidden 0.018%, logits 0.05%, cb0 argmax match vs the
  `QWEN3_TTS_DUMP_TALKER` reference). Talker stage 3.9 -> 2.4 ms/frame
  in-context; ~19% faster end-to-end on a ~190-frame greedy request.
- Integrated in `TTSTransformer::generate()` behind
  `QWEN3_TTS_USE_HIP_TALKER=1`. **Off by default.**
- **NOT in production.** `tts-engine.service` runs the toolbox-built
  `build-hip/qwen3-tts-server` (Sep 10, before the talker work) with
  only `QWEN3_TTS_USE_HIP_CODE_PRED=1`. The talker there is the plain
  ggml HIP backend.
- Built on the **host** with therock ROCm in `build-hip-host/` (see
  "Build" below). The toolbox `build-hip/` does NOT have the talker yet.

## Do NOT just flip the env var in the service. Three gates first.

### Gate 1 — mid-generation fallback (correctness on a shared GPU)

The fused talker is a cooperative kernel needing full grid residency.
The service shares the GPU with `llama-server` and `embedding-server`.
If the grid can't launch, the deadline barrier times out and
`HipTalker::run` returns false — and the current integration does
`return false` from `generate()` (tts_transformer.cpp ~3651), failing
the whole request.

The fused code predictor already handles this: on failure
`predict_codes_autoregressive` logs and falls back to the ggml path.
The talker needs the same, but mid-generation is trickier because the
fused path owns `pending_cb0` and the device-side `seen[]` set.

Required fix: on `run()` failure, fall back to `forward_step` + the
host `sample_cb0` lambda for that frame and keep going. The host must
then re-sync the device `seen[]` set with `generated_cb0_tokens` (call
`reset_repetition()` then `mark_seen()` for each token) so the penalty
stays consistent across the fused/ggml boundary. Decide: latch
`hip_talker_failed_` on first failure (simplest, one fallback frame)
or retry per frame. Recommend latch-and-fallback.

Add a test: run the fused talker while a second process saturates the
GPU and confirm the request completes via fallback rather than erroring.

### Gate 2 — seed re-audition (voice identity)

The fused talker changes the arithmetic order, so sampled codes diverge
from the ggml-talker path. Seed 3 was auditioned for **ggml talker +
fused cp**. With the fused talker the voice is a new roll.

Before pinning: run 3-4 seeds through the **fused talker + fused cp**
path on the reference voice(s) (`~/tools/reference_voices/nyx_reference.wav`),
listen, pick the keeper, pin it. Same procedure as the Phase 2 re-audition.
Do this AFTER gate 1 so you audition the shippable code.

### Gate 3 — build path

The service ExecStart points at `build-hip/` inside the `navi31-llama`
toolbox. Two options:

- **Rebuild `build-hip` in the toolbox** with the talker code (keeps the
  current infra). Use the Phase 0 toolbox recipe
  (`~/src/fedora_toolbox/Dockerfile.fedora44-therock-navi31`) plus the
  compiler-rt overlay workaround noted in `code_predictor_plan.md`.
- **Move the service to a host binary** (`build-hip-host/`). The host
  now has full ROCm and the E2E (incl. vocoder) runs natively — no
  toolbox glibc dance. Cleaner long-term, but changes the service's
  container/kill-mode assumptions (the current unit shares the toolbox
  with llama-server via `KillMode=process`). Decide deliberately.

## Enabling, once gates pass

Add `QWEN3_TTS_USE_HIP_TALKER=1` to the service ExecStart env (next to
`QWEN3_TTS_USE_HIP_CODE_PRED=1`), `systemctl --user daemon-reload &&
restart tts-engine`, and watch the journal for the
`HIP fused talker: N blocks` line and any `grid barrier timed out`
aborts under real load. Roll back by removing the env var.

## Next technical lever (after shipping, or instead of shipping now)

The fused talker sits at 2.4 ms against a ~1.3 ms bandwidth floor. The
gap is ~170 grid barriers/step and the attention phase using only 16 of
the grid's blocks. Two directions, in rough value order:

1. **Device-resident chaining.** Today each frame does a 4 KB hidden
   D2H (talker) + 4 KB H2D (cp) round-trip. `HipTalker::device_hidden()`
   already exposes the device buffer. Add a cp entry point that reads
   hidden from a device pointer (skip the host copy), with a stream
   event between the talker stream and the cp stream. Removes ~8 KB/frame
   of PCIe and one sync from the critical path.
2. **Single fused talker+cp kernel.** One cooperative launch per frame,
   one sync, cb0 and the 15 codes all produced on device. Biggest win,
   biggest change: the cp's 16-position loop and the talker's 28-layer
   step share one barrier schedule. Revisit the residency math — a larger
   cooperative grid is harder to schedule on a busy GPU, which makes
   gate 1's fallback even more important.

Cheaper follow-ups: parallelize the talker attention across more blocks
(split-K over positions); tune blocks/CU per n_past.

## Build / test / bench (host, this machine)

```bash
# configure (host therock ROCm; use clang++ directly, NOT the hipcc wrapper)
cmake -S . -B build-hip-host -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
      -DQWEN3_TTS_TIMING=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/home/sreed/tools/therock-tarball/install/lib/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/home/sreed/tools/therock-tarball/install
cmake --build build-hip-host -j8

# reference dump (first decode step: step_embd, hidden, logits, full KV cache)
QWEN3_TTS_DUMP_TALKER=/tmp/talker_ref_step.bin \
  ./build-hip-host/qwen3-tts-cli -m models/qwen3-tts-0.6b-f16.gguf \
  --vocoder models/qwen3-tts-tokenizer-f16.gguf \
  -r ~/tools/reference_voices/nyx_reference.wav -t "Hello, this is a test." \
  --temperature 0 --max-tokens 8 -o /tmp/talker_ref.wav

# parity test (greedy default; QWEN3_TTS_TEST_TEMP/TOPK for sampled smoke)
./build-hip-host/test_hip_talker --model models/qwen3-tts-0.6b-f16.gguf \
  --ref /tmp/talker_ref_step.bin --iters 30

# fused vs ggml talker, long text (amortize init; compare total generate)
LONG="...a few sentences..."
QWEN3_TTS_USE_HIP_CODE_PRED=1 ./build-hip-host/qwen3-tts-cli ... -t "$LONG" --temperature 0 --max-tokens 400
QWEN3_TTS_USE_HIP_TALKER=1 QWEN3_TTS_USE_HIP_CODE_PRED=1 ./build-hip-host/qwen3-tts-cli ... -t "$LONG" --temperature 0 --max-tokens 400
```

## Gotchas

- **Frame-count confound.** Greedy output length depends on where EOS
  lands, which changes with the arithmetic. End-to-end ms across paths
  is only comparable at similar frame counts; trust the per-frame
  talker bucket and the unit-test kernel time, not raw totals.
- **GPU must be otherwise idle** for clean timings (see the
  `drm-resident` recipe in `code_predictor_plan.md`). With
  llama-server resident the cooperative grid may not launch at all —
  which is exactly the gate-1 scenario.
- **`tensor->data` is the device pointer** for backend-allocated ggml
  tensors; that's how the fused talker shares the prefill KV cache and
  reads weights in place. Don't re-copy them.
- The `[loop wall / host tail]` line in the timing dump is diagnostic
  added this session (under `QWEN3_TTS_TIMING`); host tail is ~0.5
  ms/frame and is not the bottleneck.
