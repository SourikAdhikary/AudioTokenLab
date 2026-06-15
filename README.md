# AudioTokenLab

**Audio-token compression benchmarks for speech and voice-model infrastructure.**

AudioTokenLab measures how much discrete audio-token streams can be compressed before speech quality breaks. It is built around the practical serving question behind audio LMs, voice agents, speech-to-speech systems, and TTS:

> Can we reduce audio-token memory and latency without destroying intelligibility or speaker identity?

The current benchmark uses EnCodec 24 kHz tokens, reconstructs compressed audio, and evaluates the result with ASR WER/CER, SpeechBrain speaker similarity, reconstruction metrics, and KV-cache estimates.

## Current Result

The latest run is a 100-clip LibriSpeech `dev-clean` benchmark on Modal L4:

```text
Dataset: LibriSpeech dev-clean
Clips: 100
Speakers: 40
Chapters: 97
Tokenizer: EnCodec 24 kHz, 6 kbps target bandwidth
ASR evaluator: faster-whisper tiny.en
Speaker evaluator: SpeechBrain ECAPA
```

| Strategy | Token Reduction | Mean WER | WER 95% CI | Speaker Sim | KV Savings |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00% | 9.39% | 6.83%-12.40% | 1.000 | 0.00 MB |
| `uniform` | 49.94% | 36.72% | 31.09%-43.30% | 0.527 | 230.17 MB |
| `acoustic_salience` | 49.94% | 14.77% | 11.75%-18.13% | 0.824 | 230.17 MB |
| `energy_tuned_e4_t1_o2` | 49.94% | 14.98% | 12.24%-18.20% | 0.831 | 230.17 MB |
| `patch` | 74.91% | 99.72% | 99.29%-100.00% | 0.019 | 345.27 MB |

The important comparison is not baseline vs compressed audio. It is **uniform dropping vs salience-based dropping at the same token budget**:

- Uniform 2x frame dropping gets roughly 50% token reduction, but WER jumps to 36.72%.
- Acoustic salience keeps the same roughly 50% token reduction, but WER is 14.77%.
- The tuned energy variant has similar WER at 14.98% and the best compressed speaker similarity at 0.831.

See the full report: [REPORT.md](REPORT.md)

## Result Artifacts

Tracked, repo-visible artifacts:

- [100-clip result summary JSON](experiments/results/encodec_librispeech_asr_modal_2026-06-15.json)
- [100-clip summary chart](experiments/results/encodec_librispeech_asr_100clip_summary_chart.svg)
- [listening examples](experiments/results/encodec_librispeech_asr_100clip_listening_examples.md)
- [committed example WAVs](experiments/results/listening_examples/)

Generated full run artifacts are intentionally ignored from git and written under `modal-runs/`.

## What It Measures

AudioTokenLab reports:

- token count and token reduction
- estimated transformer KV-cache footprint and savings
- encode/decode runtime and real-time factor
- reconstruction MSE, MAE, SNR, and duration drift
- downstream ASR WER/CER with bootstrap confidence intervals
- speaker similarity against baseline reconstruction
- failure cases with transcript and audio examples

The goal is to make tradeoffs obvious: memory savings are only useful if the resulting audio remains intelligible and voice-preserving.

## How It Works

```text
audio dataset
  -> audio tokenizer
  -> compression strategy
  -> reconstruction
  -> ASR + speaker + signal evaluation
  -> CSV / JSON / HTML / chart artifacts
```

The main neural-codec benchmark uses EnCodec. The repo also includes dependency-light dummy and mu-law tokenizers for fast local tests.

## Compression Strategies

| Strategy | Description |
| --- | --- |
| `baseline` | No compression. |
| `uniform` | Keep every Nth EnCodec frame. Simple and cheap, but destructive. |
| `acoustic_salience` | Keep the frame with the strongest local RVQ-token transition inside each window, then repeat-fill the decode timeline. |
| `energy_salience` | Combine token-transition score with frame-energy/onset cues. |
| `energy_tuned_e4_t1_o2` | Tuned energy-salience variant from the 100-clip run. |
| `patch` | Average codec IDs across frame windows. Kept as a failure baseline because arithmetic over discrete codec IDs is not meaningful. |

## Installation

Create a local editable install:

```bash
python3 -m pip install -e .
```

Optional extras:

```bash
python3 -m pip install -e '.[encodec]'
python3 -m pip install -e '.[modal]'
python3 -m pip install -e '.[asr]'
python3 -m pip install -e '.[speaker]'
```

For the full Modal benchmark, you need Modal configured:

```bash
modal setup
```

## Local Usage

Dependency-free demo:

```bash
audiotokenlab profile --config experiments/configs/demo.json
audiotokenlab report runs/demo
```

Quiet-segment workload:

```bash
audiotokenlab profile --config experiments/configs/quiet_demo.json
audiotokenlab report runs/quiet_demo
```

Mu-law tokenizer baseline:

```bash
audiotokenlab profile --config experiments/configs/mulaw_demo.json
audiotokenlab report runs/mulaw_demo
```

Optional EnCodec local run:

```bash
python3 -m pip install -e '.[encodec]'
audiotokenlab profile --config experiments/configs/encodec_demo.json
audiotokenlab report runs/encodec_demo
```

## Modal Benchmarks

Small EnCodec smoke run:

```bash
modal run modal_app.py
```

Synthetic speech plus ASR smoke run:

```bash
modal run modal_app.py --speech-asr
```

Full tuned LibriSpeech benchmark:

```bash
modal run modal_app.py --librispeech-asr --max-clips 100 --strategy-set tuned
```

This downloads LibriSpeech `dev-clean` on Modal, selects clips across speakers/chapters, converts FLAC to 24 kHz mono WAV, runs EnCodec compression/reconstruction, evaluates ASR, computes speaker similarity, and writes local artifacts under `modal-runs/encodec_librispeech_asr/`.

Expected full-run artifacts:

```text
modal-runs/encodec_librispeech_asr/
  manifest.json
  metrics.csv
  dashboard.html
  asr_metrics.csv
  asr_summary.json
  speaker_metrics.csv
  speaker_summary.json
  publication_summary.json
  summary_chart.svg
  listening_examples.md
  samples/
    *.wav
```

## Repository Layout

```text
src/audiotokenlab/
  compression.py          compression strategies
  profiling.py            per-clip benchmark metrics
  asr_eval.py             WER/CER evaluation and bootstrap CIs
  speaker_eval.py         SpeechBrain speaker-similarity evaluation
  publication.py          chart and listening-example artifacts
  reporting.py            CSV/JSON/HTML reporting
  tokenizers/             dummy, mu-law, and EnCodec tokenizer adapters

experiments/
  configs/                local run configs
  results/                tracked benchmark summaries and public artifacts

modal_app.py              Modal GPU benchmark entrypoint
REPORT.md                 current benchmark report
```

## Why This Matters

Audio models pay for long audio contexts in tokens, just like text models do. Codec-token streams can be dense, and in autoregressive or transformer-style audio systems they affect:

- prefill latency
- decode latency
- KV-cache memory
- batching efficiency
- serving cost
- time-to-first-audio

Naively dropping audio tokens saves memory but can erase phonetics, timing, speaker identity, and prosody. AudioTokenLab is a measurement harness for finding better quality/cost tradeoffs.

## Research Context

AudioTokenLab is motivated by recent work on efficient audio-token modeling and long-form speech systems:

- [TLDR: Compressing Audio Tokens for Efficient Autoregressive Text-to-Speech](https://arxiv.org/abs/2606.09019)
- [Speech-XL](https://arxiv.org/abs/2602.05373)
- [LLM-Codec](https://arxiv.org/abs/2604.17852)
- [Ultra-Low Latency Streaming Speech Synthesis via Block-Wise Generation](https://arxiv.org/abs/2604.12438)
- [Building Enterprise Realtime Voice Agents from Scratch](https://arxiv.org/abs/2603.05413)

This repo is not trying to reproduce those systems end to end. It builds the measurement layer around the bottleneck they expose: audio-token cost.

## Current Limitations

- The main benchmark uses LibriSpeech `dev-clean`; broader datasets are still needed.
- ASR uses `faster-whisper tiny.en`, which is a practical evaluator but not an oracle for speech quality.
- Speaker similarity uses one pretrained embedding model.
- The current salience policies are heuristic, not learned token selectors.
- KV-cache savings are architecture estimates, not measurements from a deployed audio transformer.

## Roadmap

- [x] Local profiling pipeline
- [x] Dummy and mu-law tokenizer baselines
- [x] Optional EnCodec backend
- [x] Modal GPU benchmark path
- [x] LibriSpeech real-speech benchmark
- [x] ASR WER/CER with bootstrap confidence intervals
- [x] SpeechBrain speaker similarity
- [x] Salience-based compression baselines
- [x] Energy-salience tuning run
- [x] Public report, chart, and listening examples
- [ ] Broader datasets beyond LibriSpeech
- [ ] Stronger VAD or learned token selector
- [ ] Subjective listening study
- [ ] Integration with a real audio-token transformer serving stack

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

## License

MIT. Dataset and model licenses remain governed by their upstream providers.
