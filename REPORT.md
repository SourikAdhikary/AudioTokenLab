# AudioTokenLab Benchmark Report

## Summary

AudioTokenLab benchmarks how much EnCodec audio-token streams can be compressed before downstream speech utility breaks.

The current benchmark evaluates EnCodec 24 kHz reconstructions with `faster-whisper` and SpeechBrain ECAPA speaker embeddings on a 100-clip LibriSpeech `dev-clean` slice. The main result: simple salience-based sparse-frame retention gives the same roughly 50% token reduction as uniform frame dropping, but preserves both ASR and speaker similarity much better.

The repo now has the next-stage benchmark machinery implemented as well: broader multi-corpus manifests, VAD/selector strategies, subjective listening-study sheets, and a serving-stack report for transformer prefill/KV-cache tradeoffs. The numbers below remain the tracked LibriSpeech public result until a new broader Modal run is committed.

## Benchmark Setup

- Dataset: LibriSpeech `dev-clean`
- Source: OpenSLR SLR12, https://www.openslr.org/12/
- Clips: 100
- Speaker count: 40
- Chapter count: 97
- Audio format: mono 24 kHz WAV
- Tokenizer: EnCodec 24 kHz, 6 kbps target bandwidth
- Hardware: Modal L4 GPU
- ASR evaluator: `faster-whisper` `tiny.en`, CPU int8
- Speaker evaluator: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`
- Modal run: https://modal.com/apps/sourikadhikary/main/ap-cSbD1joos50zkMK4axaVX1
- Strategy set: tuned energy-salience ablation

Raw generated artifacts are intentionally ignored from git and live locally under:

```text
modal-runs/encodec_librispeech_asr/
```

The tracked machine-readable summary is:

```text
experiments/results/encodec_librispeech_asr_modal_2026-06-15.json
```

## Strategies

| Strategy | Description |
| --- | --- |
| `baseline` | No token compression. |
| `uniform` | Keep every second EnCodec frame. |
| `acoustic_salience` | In each 2-frame window, keep the RVQ frame with the strongest local token transition and repeat-fill the decode timeline. |
| `energy_salience` | In each 2-frame window, combine local token transition, frame energy, and onset-like energy changes before repeat-filling the decode timeline. |
| `energy_tuned_e4_t1_o2` | Tuned energy salience variant with stronger frame-energy weight. |
| `vad_salience` | Stronger speech-activity selector using normalized frame energy, short-run filtering, and hangover. Available in `--strategy-set extended`. |
| `linear_selector_v1` | Learned-selector integration point: a linear frame selector over energy, onset, transition, and speech-activity features. Available in `--strategy-set extended`. |
| `patch` | Average codec IDs across 4-frame windows. This is kept as a failure baseline because arithmetic over discrete codec IDs is not semantically meaningful. |

## Results

![Token reduction vs WER](experiments/results/encodec_librispeech_asr_100clip_summary_chart.svg)

| Strategy | Token Reduction | Mean WER | WER 95% CI | Mean CER | Speaker Sim | KV Savings | Mean SNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00% | 9.39% | 6.83%-12.40% | 4.94% | 1.000 | 0.00 MB | 7.05 dB |
| `uniform` | 49.94% | 36.72% | 31.09%-43.30% | 18.85% | 0.527 | 230.17 MB | -1.28 dB |
| `acoustic_salience` | 49.94% | 14.77% | 11.75%-18.13% | 8.00% | 0.824 | 230.17 MB | -0.11 dB |
| `energy_salience` | 49.94% | 17.23% | 13.86%-21.15% | 8.97% | 0.829 | 230.17 MB | -0.24 dB |
| `energy_tuned_e4_t1_o2` | 49.94% | 14.98% | 12.24%-18.20% | 7.61% | 0.831 | 230.17 MB | -0.23 dB |
| `patch` | 74.91% | 99.72% | 99.29%-100.00% | 97.85% | 0.019 | 345.27 MB | -6.42 dB |

## Interpretation

Uniform frame dropping is cheap, but it damages intelligibility and voice identity. On this 100-clip slice, it increases WER from 9.39% to 36.72% and drops speaker similarity to 0.527.

The salience baselines keep the same token budget as uniform dropping but preserve ASR and speaker similarity much better:

- `acoustic_salience`: 14.77% WER
- `energy_tuned_e4_t1_o2`: 14.98% WER
- `uniform`: 36.72% WER

The tuned energy variant does not beat acoustic salience on WER, but it has the best compressed-speaker similarity and the best CER among salience variants in this run. That makes it the more interesting starting point for future VAD-aware work.

The `patch` result is intentionally bad. It confirms that naive arithmetic over discrete codec IDs is a failure mode, not a viable compression method.

## Failure Cases

The generated dashboard includes worst-case transcript rows and audio controls:

```text
modal-runs/encodec_librispeech_asr/dashboard.html
```

Tracked shareable artifacts:

```text
experiments/results/encodec_librispeech_asr_100clip_summary_chart.svg
experiments/results/encodec_librispeech_asr_100clip_listening_examples.md
```

The main failure mode is not subtle: uniform frame dropping often turns words into plausible but wrong phrases. Patch averaging can collapse into empty or unrelated transcriptions. Salience methods still make word errors, especially on longer utterances, but they preserve enough local acoustic structure to stay far closer to the baseline.

## Broader-Speech Smoke Validation

A tiny Modal smoke run validates the next-stage pipeline across three sources:

- LibriSpeech `dev-clean`
- MInDS-14 `en-US`
- FLEURS `en_us`

Smoke run details:

- Modal run: https://modal.com/apps/sourikadhikary/main/ap-wLrk8GXPvKU9w02oGqYWb9
- Clips: 3
- Strategies: 7
- Reconstructed/evaluated samples: 21
- Strategy set: `extended`

Tracked smoke artifacts:

```text
experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_publication_summary.json
experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_serving_stack_report.md
experiments/results/encodec_broader_speech_asr_smoke_2026-06-15_listening_study.csv
```

This smoke run is not a statistically meaningful benchmark. It proves the broader dataset path, VAD/selector strategies, subjective listening sheet, serving-stack report, and CUDA PyTorch transformer microbenchmark all execute end to end on Modal.

## Next-Stage Workflows

Broader speech benchmark:

```bash
modal run modal_app.py --broader-speech-asr --max-clips-per-source 4 --strategy-set extended
```

This combines LibriSpeech with public Hugging Face speech corpora such as MInDS-14 and FLEURS, subject to upstream access and license constraints. The corpus builder is source-configurable for TED-LIUM, Common Voice, VoxPopuli, or internal manifests when access is available. It emits the same ASR, speaker, publication, listening, and serving artifacts as the LibriSpeech path.

Serving-stack benchmark:

```bash
modal run modal_app.py --broader-speech-asr --max-clips-per-source 4 --strategy-set extended --serving-microbench
```

The serving report writes:

```text
serving_stack_report.json
serving_stack_report.md
```

It estimates transformer prefill attention-work ratio, decode KV-read reduction, and decode KV-read MB saved. With `--serving-microbench`, it also runs a reference PyTorch transformer layer on representative token lengths.

Subjective listening artifacts:

```text
listening_study.csv
listening_study.md
listening_study.json
```

The CSV is an anonymized rating sheet for MOS, intelligibility, speaker match, and artifact notes. It is generated from the existing ASR and speaker outputs, so it can be shared with listeners without changing the benchmark pipeline.

## Launch Summary

Short version:

> I built AudioTokenLab, a benchmark for audio-token compression. On a 100-clip LibriSpeech EnCodec run, naive 2x frame dropping cut tokens by 50% but pushed WER to 36.7%. A simple salience policy kept the same 50% token reduction while cutting WER to 14.8% and preserving speaker similarity much better.

Numbers to mention:

- 100 real speech clips
- 800 reconstructed samples
- Modal L4 run
- EnCodec 24 kHz tokens
- `faster-whisper` WER/CER
- SpeechBrain ECAPA speaker similarity
- 49.96% token reduction
- 36.72% WER for uniform dropping vs 14.77% WER for acoustic salience
- best tuned energy variant: `energy_tuned_e4_t1_o2`

## Current Limitations

- 100 clips is a stronger benchmark than the smoke run, but still not a publication-grade estimate.
- `faster-whisper` `tiny.en` is a convenient evaluator, not an oracle for speech quality.
- Speaker similarity is measured with one pretrained embedding model; subjective voice quality and prosody are outside the v1 metric scope.
- `vad_salience` is stronger than raw frame energy, but it is still deterministic signal processing rather than a trained VAD.
- `linear_selector_v1` is a selector hook; default weights are hand-set unless a run explicitly reports trained weights.
- The serving stack currently benchmarks transformer-shaped token workloads, not a production voice-agent model with live traffic.

## Next Research Steps

1. Run and publish the broader multi-corpus Modal benchmark.
2. Train selector weights against ASR/speaker preservation instead of using the default linear weights.
3. Collect subjective listening ratings for the generated study sheet.
4. Swap the reference PyTorch transformer microbenchmark for a selected production-grade audio-token model.
