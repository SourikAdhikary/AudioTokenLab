# AudioTokenLab

**A benchmark lab for the token economics of audio language models.**

Modern speech and audio systems increasingly work like language models. They turn waveform audio into discrete codec tokens, run transformer-style models over those tokens, and decode the result back into sound.

That design unlocked impressive TTS, speech-to-speech, music generation, dubbing, and voice-agent systems. It also created a new infrastructure problem:

> Audio tokens are expensive.

Audio streams are much denser than text. A short spoken sentence can become hundreds or thousands of codec tokens. For autoregressive audio models, those tokens drive prefill cost, decode latency, KV-cache memory, GPU utilization, and time-to-first-audio.

AudioTokenLab is a systems project for measuring that cost and testing how far audio-token streams can be compressed before quality breaks.

## The Problem

Text model infra has a mature vocabulary:

- tokens per request
- prefill latency
- decode throughput
- KV-cache memory
- batching efficiency
- cost per million tokens

Audio model infra needs the same rigor, but the units are messier.

Audio tokens carry phonetics, speaker identity, rhythm, noise, prosody, and acoustic detail. Reducing token count may save memory and latency, but it can also damage:

- intelligibility
- speaker similarity
- timing
- emotion and prosody
- downstream ASR accuracy
- reconstruction quality

AudioTokenLab asks a concrete question:

> Given an audio tokenizer and workload, what is the best quality/cost tradeoff we can get from token compression?

## What This Project Will Build

AudioTokenLab is planned as a reproducible profiling and compression pipeline:

```text
audio dataset
  -> neural codec tokenizer
  -> token profiler
  -> compression strategy
  -> reconstruction
  -> quality evaluation
  -> cost and latency report
```

The first version will focus on speech because speech gives measurable automatic signals: transcript preservation, character/word error rate, speaker similarity, duration drift, and reconstruction quality.

## What It Measures

For each audio workload, AudioTokenLab will report:

- audio duration and sample rate
- codec tokens per second
- codebook and frame statistics
- encode/decode runtime
- real-time factor
- estimated KV-cache footprint
- estimated transformer serving cost
- compression ratio
- ASR WER/CER before and after compression
- speaker similarity before and after compression
- reconstruction quality proxies
- MAE and SNR signal metrics
- failure cases where compression visibly breaks the audio

The goal is not to hide tradeoffs. The goal is to make them obvious.

## Compression Experiments

Initial strategies:

- **Baseline**: no compression
- **Uniform compression**: simple fixed-rate token reduction
- **Silence-aware compression**: compress low-information regions more aggressively
- **Patch compression**: group consecutive codec tokens into larger units
- **Acoustic salience compression**: keep RVQ frames with the strongest local token transitions, then repeat-fill the decode timeline

The project starts with simple, inspectable baselines before introducing learned compression. That keeps the benchmark honest and makes every improvement easier to interpret.

## Why This Matters

Audio-token compression is becoming relevant across several active areas:

- real-time TTS
- speech-to-speech models
- voice agents
- long-form meeting and call understanding
- multilingual dubbing
- audio generation
- music generation
- video generation with synchronized audio
- edge and wearable voice interfaces

If audio models continue to scale like language models, token length becomes an infra problem. Better profiling and compression can reduce serving cost, improve latency, and make longer audio contexts practical.

## Research Context

AudioTokenLab is motivated by recent work on efficient audio-token modeling and long-form speech systems:

- [TLDR: Compressing Audio Tokens for Efficient Autoregressive Text-to-Speech](https://arxiv.org/abs/2606.09019) explores patch-level codec-token modeling for faster AR-TTS and lower KV-cache usage.
- [Speech-XL](https://arxiv.org/abs/2602.05373) studies long-form speech understanding using speech summarization tokens and KV sparsification.
- [LLM-Codec](https://arxiv.org/abs/2604.17852) argues that reconstruction-oriented audio codecs are not necessarily ideal tokenizers for language-model objectives.
- [Ultra-Low Latency Streaming Speech Synthesis via Block-Wise Generation](https://arxiv.org/abs/2604.12438) shows how codec-space design affects streaming TTS latency.
- [Building Enterprise Realtime Voice Agents from Scratch](https://arxiv.org/abs/2603.05413) highlights why production voice systems are dominated by streaming, pipelining, and latency constraints.

AudioTokenLab is not trying to reproduce those systems end to end. It is building the measurement layer around the bottleneck they expose.

## Example Output

A completed run should produce a report like:

```text
Tokenizer: encodec_24khz
Dataset: demo_speech_30

Strategy          Token Reduction   WER Delta   Speaker Sim   Est. KV Savings
baseline          0.0%              0.0         1.00          0.0%
uniform_2x        50.0%             +8.4        0.82          50.0%
silence_aware     31.2%             +1.7        0.94          31.2%
patch_4           58.5%             +3.1        0.91          58.5%
adaptive_patch    47.8%             +1.2        0.95          47.8%
```

Numbers above are illustrative. The project will only claim results backed by generated artifacts.

## Planned CLI

Install locally in editable mode:

```bash
python3 -m pip install -e .
```

Run the dependency-free demo profile:

```bash
audiotokenlab profile --config experiments/configs/demo.json
audiotokenlab report runs/demo
```

Run a quiet-segment workload that exercises silence-aware compression:

```bash
audiotokenlab profile --config experiments/configs/quiet_demo.json
audiotokenlab report runs/quiet_demo
```

Run the first real audio quantizer baseline:

```bash
audiotokenlab profile --config experiments/configs/mulaw_demo.json
audiotokenlab report runs/mulaw_demo
```

Run the optional EnCodec backend after installing the extra dependencies:

```bash
python3 -m pip install -e '.[encodec]'
audiotokenlab profile --config experiments/configs/encodec_demo.json
audiotokenlab report runs/encodec_demo
```

Run the EnCodec benchmark on Modal:

```bash
python3 -m pip install -e '.[modal]'
modal run modal_app.py
```

Modal artifacts are written locally under `modal-runs/` by the local entrypoint.

Run the Modal speech+ASR smoke benchmark:

```bash
modal run modal_app.py --speech-asr
```

This synthesizes two tiny speech clips with `espeak-ng`, runs EnCodec compression, transcribes reconstructed samples with `faster-whisper`, and writes `asr_metrics.csv` plus `asr_summary.json`.

Run the real-speech LibriSpeech ASR benchmark:

```bash
modal run modal_app.py --librispeech-asr --max-clips 4
```

This downloads a tiny slice from LibriSpeech `dev-clean` on Modal, converts selected FLAC clips to 24 kHz WAV, runs EnCodec compression, and evaluates reconstructed samples with `faster-whisper`.

First real-speech smoke result:

```text
Dataset: LibriSpeech dev-clean, 4 clips
Baseline WER: 3.45%
Uniform WER: 52.26% at ~49.93% token reduction
Acoustic salience WER: 19.40% at ~49.93% token reduction
Patch WER: 100.00% at ~74.84% token reduction
```

Expected artifacts:

```text
runs/demo/
  manifest.json
  metrics.csv
  dashboard.html
  samples/
    synthetic_000__baseline.wav
```

## Project Roadmap

- [x] Local pipeline with a dummy tokenizer
- [x] Quiet-segment synthetic workload for silence-aware compression
- [x] Dependency-free mu-law audio tokenizer baseline
- [x] Reconstructed WAV sample artifacts
- [x] Optional EnCodec backend adapter
- [x] Verified EnCodec benchmark run
- [x] Token profiling and KV-cache estimation
- [x] Baseline compression strategies
- [x] Speech reconstruction evaluation
- [x] ASR-based WER/CER regression checks
- [x] Acoustic salience sparse-frame baseline
- [ ] Speaker similarity checks
- [x] HTML dashboard
- [x] Modal GPU benchmark run
- [x] ASR-based WER/CER smoke run
- [x] Real-speech LibriSpeech smoke run
- [ ] Public benchmark report

## Design Principles

- **Measure first**: compression without metrics is guesswork.
- **Keep baselines simple**: naive methods reveal what the hard part actually is.
- **Report failures**: bad reconstructions and WER spikes are part of the result.
- **Avoid fake SOTA claims**: project claims must come from reproducible runs.
- **Optimize for systems insight**: the main output is understanding latency, memory, cost, and quality tradeoffs.

## Status

AudioTokenLab is in early development. The current scaffold includes a dependency-free profiling pipeline with a dummy tokenizer, compression baselines, KV-cache estimates, CSV/JSON/HTML report artifacts, and unit tests.

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

## License

License to be decided before the first code release.
