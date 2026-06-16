# AudioTokenLab

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Benchmark: Modal L4](https://img.shields.io/badge/benchmark-Modal%20L4-7c3aed)](REPORT.md)
[![Tokenizer: EnCodec](https://img.shields.io/badge/tokenizer-EnCodec-orange)](https://github.com/facebookresearch/encodec)
[![Datasets: LibriSpeech+](https://img.shields.io/badge/datasets-LibriSpeech%20%2B%20HF%20speech-blue)](https://www.openslr.org/12/)
[![Run: 100 clips](https://img.shields.io/badge/run-100%20clips%20%7C%20800%20samples-111827)](experiments/results/encodec_librispeech_asr_modal_2026-06-15.json)
[![Broader Run: 75 clips](https://img.shields.io/badge/broader-75%20clips%20%7C%20525%20samples-0f766e)](experiments/results/encodec_broader_speech_asr_modal_2026-06-16_publication_summary.json)

**Audio-token compression benchmarks for speech and voice-model infrastructure.**

AudioTokenLab measures how much discrete audio-token streams can be compressed before speech quality breaks. It is built around the practical serving question behind audio LMs, voice agents, speech-to-speech systems, and TTS:

> Can we reduce audio-token memory and latency without destroying intelligibility or speaker identity?

The current benchmark uses EnCodec 24 kHz tokens, reconstructs compressed audio, and evaluates the result with ASR WER/CER, SpeechBrain speaker similarity, reconstruction metrics, and KV-cache estimates.

The repo now also includes the next-stage research hooks:

- broader multi-corpus speech manifests beyond LibriSpeech
- VAD-aware and linear learned-selector token retention strategies
- subjective listening-study sheets for human ratings
- serving-stack reports for transformer prefill/KV-cache tradeoffs, with an optional PyTorch microbenchmark

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

## Broader Speech Result

The broader benchmark runs the same EnCodec pipeline on 75 clips across three sources:

```text
LibriSpeech dev-clean: 25 clips
MInDS-14 en-US:        25 clips
FLEURS en_us:          25 clips
Strategies:            7
Evaluated samples:     525
Modal run:             ap-GvQq49rJPkHf3SMC3joT5H
```

| Strategy | Token Reduction | Mean WER | Speaker Sim | KV Savings |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0.00% | 36.05% | 1.000 | 0.00 MB |
| `uniform` | 49.95% | 60.61% | 0.460 | 232.97 MB |
| `acoustic_salience` | 49.95% | 49.39% | 0.792 | 232.97 MB |
| `energy_salience` | 49.95% | 49.05% | 0.796 | 232.97 MB |
| `linear_selector_v1` | 49.95% | 50.38% | 0.792 | 232.97 MB |
| `vad_salience` | 49.95% | 51.15% | 0.792 | 232.97 MB |
| `patch` | 74.92% | 99.55% | 0.043 | 349.47 MB |

This mixed-domain run is harder than LibriSpeech for `tiny.en`, so absolute WER is higher. The important signal is still the same-budget comparison: `energy_salience` keeps roughly 50% token reduction while improving WER by 11.56 percentage points over uniform dropping and preserving much more speaker identity.

## Result Artifacts

Tracked, repo-visible artifacts:

- [100-clip result summary JSON](experiments/results/encodec_librispeech_asr_modal_2026-06-15.json)
- [100-clip summary chart](experiments/results/encodec_librispeech_asr_100clip_summary_chart.svg)
- [listening examples](experiments/results/encodec_librispeech_asr_100clip_listening_examples.md)
- [committed example WAVs](experiments/results/listening_examples/)
- [75-clip broader publication summary](experiments/results/encodec_broader_speech_asr_modal_2026-06-16_publication_summary.json)
- [75-clip broader summary chart](experiments/results/encodec_broader_speech_asr_modal_2026-06-16_summary_chart.svg)
- [75-clip broader serving report](experiments/results/encodec_broader_speech_asr_modal_2026-06-16_serving_stack_report.md)
- [75-clip broader listening-study sheet](experiments/results/encodec_broader_speech_asr_modal_2026-06-16_listening_study.csv)
- [broader-speech smoke publication summary](experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_publication_summary.json)
- [broader-speech smoke serving report](experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_serving_stack_report.md)
- [broader-speech smoke listening-study sheet](experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_listening_study.csv)

Generated full run artifacts are intentionally ignored from git and written under `modal-runs/`.

The broader-speech smoke artifact is intentionally small: 3 clips across LibriSpeech, MInDS-14, and FLEURS; 7 strategies; 21 reconstructed/evaluated samples. Its purpose is to verify the pipeline and artifact generation.

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
| `vad_salience` | Uses frame-energy speech activity, short-run filtering, and hangover to keep likely speech/onset frames. |
| `linear_selector_v1` | Linear frame selector hook over energy, onset, transition, and speech-activity features. Weights can be swapped for trained weights. |
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
python3 -m pip install -e '.[datasets]'
python3 -m pip install -e '.[serving]'
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

Broader speech benchmark:

```bash
modal run modal_app.py --broader-speech-asr --max-clips-per-source 25 --strategy-set extended --serving-microbench
```

This builds one manifest from LibriSpeech plus public Hugging Face speech corpora such as MInDS-14 and FLEURS, then runs the same EnCodec + ASR + speaker pipeline. The corpus builder is source-configurable, so you can point it at TED-LIUM, Common Voice, VoxPopuli, or internal manifests when access is available. Upstream dataset access and licenses remain governed by each provider.

Small smoke variant:

```bash
modal run modal_app.py --broader-speech-asr --max-clips-per-source 1 --strategy-set extended
```

The serving report consumes `metrics.csv` and writes:

```text
serving_stack_report.json
serving_stack_report.md
```

It estimates transformer prefill attention work, decode KV-read reduction, and KV-cache savings. With `--serving-microbench`, it also runs a reference PyTorch transformer layer on representative token lengths.

Subjective listening artifacts are generated with publication artifacts:

```text
listening_study.csv
listening_study.md
listening_study.json
```

The CSV is an anonymized rating sheet for MOS, intelligibility, speaker match, and artifact notes.

## Repository Layout

```text
src/audiotokenlab/
  compression.py          compression strategies
  profiling.py            per-clip benchmark metrics
  asr_eval.py             WER/CER evaluation and bootstrap CIs
  speaker_eval.py         SpeechBrain speaker-similarity evaluation
  publication.py          chart and listening-example artifacts
  listening_study.py      subjective rating sheets
  serving.py              transformer-serving estimates and optional torch microbenchmarks
  corpora.py              broader speech dataset manifest builders
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

- The tracked headline result is still LibriSpeech `dev-clean`; the broader multi-corpus result is a stronger cross-domain check, but still uses `faster-whisper tiny.en`, so absolute WER should be interpreted as evaluator stress rather than human intelligibility.
- ASR uses `faster-whisper tiny.en`, which is a practical evaluator but not an oracle for speech quality.
- Speaker similarity uses one pretrained embedding model.
- `linear_selector_v1` is a selector hook with default hand-set weights; trained weights should be reported with their training data and objective.
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

---

## License

This project is released under the [MIT License](LICENSE).

Dataset and model licenses remain governed by their upstream providers.
