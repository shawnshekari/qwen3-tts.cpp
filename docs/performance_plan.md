# Performance Optimization Plan

## Goals
- Target: RTF < 0.5x (2x faster than real-time) for 0.6B model
- Hardware: RX 7900 XTX (Vulkan), 16-core CPU, 24 GB RAM
- Use case: SillyTavern / Open-LLM-VTuber (low latency, conversational)

## Current Baseline (Jun 19, 2026)

| Model | Audio | Time | RTF | Notes |
|-------|-------|------|-----|-------|
| 0.6B F16 | 3.9s | 3222ms | 0.83x | Voice cloning (nyx_voice) |
| 0.6B F16 | 1.9s | 1557ms | 0.84x | Voice cloning (amy_voice) |
| 1.7B Q8_0 | 3.7s | 11033ms | 3.00x | Voice cloning (nyx_voice) |
| 1.7B Q8_0 | 2.4s | 7126ms | 3.00x | Voice cloning (amy_voice) |

## Phase 1: Baseline & Quick Wins

### Task 1.1: Capture Vulkan baseline timings ✅
- [x] Build with `-DQWEN3_TTS_TIMING=ON`
- [x] Run 0.6B F16 with voice cloning (nyx_voice, amy_voice)
- [x] Record: prefill, talker, code predictor, vocoder, total
- [x] Save results to this file

**Detailed breakdown (0.6B F16, nyx_voice, 3.66s audio):**
| Stage | Time | % |
|-------|------|---|
| Tokenization | 1ms | 0% |
| Speaker encode | 0ms | 0% (cached) |
| Code generation | 922ms | 30% |
| - Prefill | 40ms | 1% |
| - Talker | 211ms | 7% |
| - Code predictor | 601ms | 20% |
| Vocoder decode | 2149ms | 70% |
| **Total** | **3072ms** | **RTF 0.84x** |

**Detailed breakdown (1.7B Q8_0, nyx_voice, 3.42s audio):**
| Stage | Time | % |
|-------|------|---|
| Tokenization | 1ms | 0% |
| Speaker encode | 0ms | 0% (cached) |
| Code generation | 8304ms | 80% |
| - Prefill | 194ms | 2% |
| - Talker | 4682ms | 45% |
| - Code predictor | 3274ms | 32% |
| Vocoder decode | 2056ms | 20% |
| **Total** | **10362ms** | **RTF 3.03x** |

**Key insight:** For 0.6B, **vocoder is 70%** of time. For 1.7B, **talker is 45%** and **code predictor is 32%**.

### Task 1.2: Convert 0.6B to Q8_0
- [ ] Use khimaros' quantization script from source checkpoint
- [ ] Convert `models/Qwen3-TTS-12Hz-0.6B-Base` → `qwen3-tts-0.6b-q8_0.gguf`
- [ ] Test: same curl requests, compare speed vs F16
- [ ] Generate side-by-side WAVs (F16 vs Q8_0) for quality comparison
- [ ] **Quality test needed:** listen to output WAVs, compare F16 vs Q8_0

### Task 1.3: Speaker embedding cache (disk-based)
- [ ] Add `--voice-cache-dir` CLI flag (default: `~/.qwen3-tts-cache/`)
- [ ] Server loads cached embeddings on startup from disk
- [ ] Pre-computed embeddings stored as `.spk` files (~4 KB each)
- [ ] New voices get encoded and cached automatically
- [ ] Persists across server restarts
- [ ] Update `docs/custom_voice_setup.md` with new workflow
- [ ] Test: restart server, verify cached voices work without re-encoding

## Phase 2: Streaming Vocoder

### Task 2.1: Transformer frame callback
- [ ] Add `frame_emit_fn` to `TTSTransformer::generate()`
- [ ] Fire callback after each frame's 16 codebooks are generated
- [ ] Unit test: callback fires once per frame with 16 codes

### Task 2.2: Decoder stream state
- [ ] Implement `AudioTokenizerDecoderStream` class in `src/audio_tokenizer_decoder.{h,cpp}`
- [ ] KV cache for pre-tfm layers (8 layers, ~32 MiB)
- [ ] Conv ring buffers (15 rings, ~265 KiB)
- [ ] Unit test: chunked decode matches one-shot decode (bit-exact or < 1e-5 max abs diff)

### Task 2.3: Server SSE integration
- [ ] Wire streaming to `stream_format=="sse"` path in `src/server.cpp`
- [ ] Flush PCM chunks as they're decoded via HTTP data sink
- [ ] Add `--streaming-batch-size` CLI flag (default 0 = off)
- [ ] Test: curl with streaming, measure time-to-first-audio

### Task 2.4: Parity test
- [ ] Implement `tests/test_streaming_parity.cpp`
- [ ] Compare streaming vs one-shot at batch sizes 1, 4, 8, 16, 32
- [ ] Pass criterion: max abs diff < 1e-5 PCM float

## Phase 3: Micro-optimizations

### Task 3.1: Snake activation broadcast
- [ ] Profile vocoder with timing instrumentation
- [ ] Replace `ggml_reshape_3d` + `ggml_repeat` with pre-broadcast alpha/beta tensors
- [ ] Profile before/after vocoder time

### Task 3.2: Batch processing
- [ ] Process multiple texts with shared speaker embedding
- [ ] Useful for SillyTavern multi-utterance scenarios

## Results Log

| Date | Model | Audio | Time | RTF | Notes |
|------|-------|-------|------|-----|-------|
| Jun 19 | 0.6B F16 | 3.66s | 3072ms | 0.84x | Baseline (nyx_voice) |
| Jun 19 | 0.6B F16 | 1.98s | 1637ms | 0.83x | Baseline (amy_voice) |
| Jun 19 | 1.7B Q8_0 | 3.42s | 10362ms | 3.03x | Baseline (nyx_voice) |
| Jun 19 | 1.7B Q8_0 | 1.98s | 5967ms | 3.02x | Baseline (amy_voice) |

## Optimization Priority (based on timing data)

For **0.6B F16** (primary use case):
1. **Vocoder** (70%) - streaming decode, snake activation
2. **Code predictor** (20%) - MTP optimization, quantization
3. **Talker** (7%) - lower priority

For **1.7B Q8_0** (quality use case):
1. **Talker** (45%) - flash attention, kernel fusion
2. **Code predictor** (32%) - same as above
3. **Vocoder** (20%) - streaming decode
