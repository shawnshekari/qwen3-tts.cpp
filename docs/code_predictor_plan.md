# Code Predictor Speed-Up Plan

Working notes for cutting per-frame latency on a discrete AMD GPU. Written
2026-09-10 after a profiling session on an RX 7900 XTX (RADV, Vulkan backend,
0.6B F16). Companion to `performance_plan.md`, which holds the Strix Halo
baseline.

## Where we are

After the two dGPU vocoder fixes (`b11fd22`, `cebfbd3`) the engine runs at
**~17 ms/frame, RTF 0.27** on the XTX (8.4 s of audio in 2.25 s one-shot,
2.42 s streaming with `stream_batch_size: 8`).

| stage | per frame | notes |
|---|---|---|
| talker (28 layers) | 4.5 ms | bandwidth floor ~1.3 ms (1.2 GB F16 @ ~960 GB/s) |
| code predictor (14 steps) | **12 ms** | 48% of the frame — the target |
| vocoder | ~2 ms | all on Vulkan now |
| graph build + `sched_alloc` | ~2 ms | the 14 step graphs are shape-identical every frame |

The same stages measured on Strix Halo in June (`performance_plan.md`) give
the *same* talker/code-predictor times despite 4x less memory bandwidth.
This model is launch-latency-bound, not bandwidth-bound: lower-precision
weights (Q8/FP4) buy almost nothing on either machine.

## Anatomy of one code-predictor step

Measured with `GGML_VK_PERF_LOGGER=1` (GPU timestamps):

- **128 kernel launches** across 21 distinct op shapes
- **~760 us of GPU time** — and ~760 us of wall time. The GPU is "busy"
  for the whole step; the CPU round-trip (8 KB logits readback, top-k
  sampling, four tiny uploads) is essentially free.
- ~170 MB of weights streamed per step. At ~960 GB/s that is a **~180 us
  floor**. The remaining ~580 us is 128 x ~4.5 us of fixed per-dispatch
  cost (barrier, cache flush, dispatch latency).

So the expensive thing is not "trips to the CPU"; it is 128 separate
dispatches with a pipeline drain between each. `hipGraph`/`GGML_HIP_GRAPHS`
only removes CPU-side submission cost and would help maybe 5-10%. The lever
is **fewer dispatches** — one per step, or one per frame.

Two free findings from the same profile:

- The code predictor's down-projection is stored **F32** in the GGUF
  (`MUL_MAT_VEC f32 m=1024 k=3072`): 63 of the 170 MB per step read at
  double width. Converting that tensor to F16 is worth ~5%.
- The mask is rebuilt on the host and uploaded every step; it only depends
  on `n_past`.

The loop is `TTSTransformer::predict_codes_autoregressive` in
`src/tts_transformer.cpp`: a 2-token prefill graph, then 14x
{`build_code_pred_step_graph`, `ggml_backend_sched_alloc_graph`, 4x
`ggml_backend_tensor_set`, `ggml_backend_sched_graph_compute`,
`ggml_backend_tensor_get(logits)`, CPU top-k sample, `sched_reset`}.

The CoreML path (`predict_codes_autoregressive_coreml`, gated by
`QWEN3_TTS_USE_COREML`) is the existing seam for "alternate backend for the
code predictor stage only". A HIP implementation slots in the same way,
with the talker and vocoder staying on Vulkan; the 4 KB hidden state and
the 15 output codes cross via host memory.

## Phases

### Phase 0 — HIP build, measure the dispatch floor (hours)

Build in the `navi31-llama` toolbox (ROCm 10.1, hipcc; llama.cpp there was
built with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, see
`~/src/fedora_toolbox/Dockerfile.fedora44-therock-navi31`):

```bash
toolbox run --container navi31-llama bash -c '
  cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
        -DQWEN3_TTS_TIMING=ON -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build-hip -j'
```

Run on a spare port with `-V`, then again with `GGML_HIP_GRAPHS=ON`, and
compare `Steps (14)` ms/frame against Vulkan. The question it answers: is
ROCm's per-dispatch fixed cost lower than RADV's ~4.5 us? If it is ~2 us the
plain HIP build is already a ~1.5x win on this stage, and it confirms HIP as
the home for Phase 2.

Risk: the vendored ggml (`d4fcfe88`, April 2026) may not compile against a
ROCm 10.1 nightly.

### Phase 1 — ggml-level cleanups (about a day, ~15%, backend-agnostic)

1. Build and allocate the prefill graph and the 14 step graphs once, keep
   them across frames, and only `tensor_set` inputs per step.
2. Precompute the 16 masks (one per `n_past`) at init.
3. Convert the F32 down-projection to F16 in the GGUF (or cast at load).

### Phase 2 — fused cooperative HIP kernel (1-2 weeks)

One cooperative-groups kernel that runs an entire step with `grid.sync()`
between phases instead of 128 dispatches, and loops all 14 steps inside a
single launch:

- shapes: 5 layers, hidden 1024, 16 query heads / 8 KV heads x 128,
  intermediate 3072, KV cache <= 16 positions, 15 output heads x vocab 2048
- per step: rmsnorm -> qkv matvec -> rope -> attention over <= 16 keys ->
  o-proj + residual -> rmsnorm -> gate/up matvec -> silu*mul -> down +
  residual, x5; head matvec; temperature/top-k/softmax sampling on-GPU
  with a seeded counter-based RNG
- matvecs partitioned across workgroups by output rows; the small norms
  and softmax done redundantly per block

Estimate: ~30 grid syncs per step at ~5 us plus ~250 us of weight
streaming -> **~400 us/step, 12 ms -> ~5-6 ms per frame**, and the 2 ms of
graph overhead disappears. Frame total roughly 17 -> 9 ms. The same
treatment for the talker (28 layers, 1.3 ms floor vs 4.5 ms today) is the
follow-on.

This needs HIP specifically: Vulkan compute has no grid-wide sync, so a
GLSL version would still be ~10 dispatches per step.

## Constraints

- The engine stays on this PC and keeps serving the `:8080` OpenAI-style
  API; other clients (Open-LLM-Vtuber) hit it directly.
- Any change to arithmetic order — Phases 1 and 2 included — changes the
  sampled codes for a given seed, which re-rolls the cloned voice's
  character. Plan for **one** re-audition of seeds afterwards and pin new
  ones; do not chase bit-exact parity.
- Benchmarks on a spare port are only valid with the live `tts-engine`
  stopped: with llama-server resident the card is otherwise full, the test
  instance spills into GTT, and everything measures 2x slow. Check with
  `grep drm-resident /proc/<pid>/fdinfo/*`.

## Profiling recipes (what actually worked)

- **Where does each op run?** `GGML_SCHED_DEBUG=1` prints one `## SPLIT #n:
  <backend>` line per split; `=2` adds every node. This is how the vocoder
  was caught running on CPU while the load log said "backend: Vulkan0" —
  the load message reports the *compute* backend, not where the weights
  landed. A split labelled `CPU # 0 inputs` covering a whole graph means
  the weights are in a CPU buffer.
- **Per-op GPU time:** `GGML_VK_PERF_LOGGER=1` prints a per-graph block
  with `<op> <shape>: N x t us = total`. It flushes a block when the *next*
  graph runs, so always make two requests, and the last graph of a run is
  never printed. Kernel count per graph = the sum of the `N x` multipliers.
- **Stage wall times:** start the server with `-V`; each request then ends
  with `Code generation / Vocoder decode / Total` in ms plus RTF. The
  `QWEN3_TTS_TIMING=ON` build (default in `build-vulkan.sh`) adds the
  per-frame talker / code-predictor / prefill breakdown.
- **Is it really in VRAM?** `cat /proc/<pid>/fdinfo/* | grep drm-resident`
  sums to VRAM vs GTT residency for the process. The healthy load on the
  XTX is ~1.7-2.4 GB VRAM and ~0.8-1.0 GB GTT (staging); if GTT climbs past
  that after a request, compute buffers spilled and timings are 2x off.
- **Output identity:** same text + same seed is bit-exact run to run on a
  given build, so `cmp` on the WAVs is a valid regression test for "did
  the codes change". When they differ only by vocoder numerics, SNR is
  ~40 dB (GPU vs CPU vocoder measured 41 dB); when the sampled codes
  differ the length changes.
- `pkill -f 'qwen3-tts-server.*-p 8081'` also matches the shell that
  launched it; use `pkill -f '^build/qwen3-tts-server'`.

## Other observations

- Both dGPU bugs came from the fork being developed on Strix Halo (an
  iGPU): `load_tensor_data_from_file(..., GGML_BACKEND_DEVICE_TYPE_IGPU)`
  fails on a discrete card and silently fell to CPU (`b11fd22`), and
  ggml-vulkan's `CONV_TRANSPOSE_1D` only accepts F32 weights so the F16
  ones in the tokenizer GGUF bounced six convs to CPU (`cebfbd3`). When
  adding ops, check `ggml_backend_vk_supports_op` for type restrictions —
  several Vulkan kernels are F32-only.
- Streaming (`stream_format: "audio"`, `stream_batch_size: 8`) runs the
  vocoder once per batch, so vocoder cost is amplified in streaming mode:
  before the conv fix streaming took 4.1 s vs 2.6 s one-shot; after, 2.42
  vs 2.25 s. First audio arrives in under a second.
- `-j` (compute threads) made no measurable difference once nothing ran
  on the CPU.
- Voices registered via `POST /v1/audio/voices` without `ref_text` use the
  speaker embedding only (`[icl] ref_frames=0`). Supplying `ref_text`
  enables ICL cloning, which is a quality lever untested so far — and
  would re-roll the voice like everything else.

## Things already ruled out

- **Bumping ggml to v0.23.0** (branch `ggml-bump`): vocoder 232 -> 141 ms,
  but talker and code predictor +20-25% per frame (net +15% per second of
  audio), and it re-rolls the voice. Re-test against a future release; the
  gate is per-second-of-audio cost *and* unchanged output for seed 42.
- **Code predictor on CPU** (5800X3D, 8 threads): 3.5 ms/step, ~4x slower.
- **Q8/FP4 weights or FP4 training**: bandwidth is ~15% of the frame;
  Strix Halo numbers with 4x less bandwidth are identical. Not the
  bottleneck on either machine.
- **Moving upstream**: khimaros has been dormant since 2026-06-16 (PRs #3
  and #4 unreviewed since June 19); predict-woo has no server or streaming
  and issue #20 (the vocoder-on-CPU bug) has been open since March.
  `shawnshekari/qwen3-tts.cpp` is the maintained copy.
