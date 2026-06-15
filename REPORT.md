# AudioTokenLab Benchmark Report

## Summary

AudioTokenLab benchmarks how much EnCodec audio-token streams can be compressed before downstream speech utility breaks.

The current benchmark evaluates EnCodec 24 kHz reconstructions with `faster-whisper` and SpeechBrain ECAPA speaker embeddings on a small but real LibriSpeech `dev-clean` slice. The main result: simple salience-based sparse-frame retention gives the same roughly 50% token reduction as uniform frame dropping, but preserves both ASR and speaker similarity much better.

## Benchmark Setup

- Dataset: LibriSpeech `dev-clean`
- Source: OpenSLR SLR12, https://www.openslr.org/12/
- Clips: 24
- Speaker count: 10
- Chapter count: 24
- Audio format: mono 24 kHz WAV
- Tokenizer: EnCodec 24 kHz, 6 kbps target bandwidth
- Hardware: Modal L4 GPU
- ASR evaluator: `faster-whisper` `tiny.en`, CPU int8
- Speaker evaluator: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`
- Modal run: https://modal.com/apps/sourikadhikary/main/ap-pt00u1Xv5YmXCmhyEXaKmO

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
| `patch` | Average codec IDs across 4-frame windows. This is kept as a failure baseline because arithmetic over discrete codec IDs is not semantically meaningful. |

## Results

| Strategy | Token Reduction | Mean WER | WER 95% CI | Mean CER | Speaker Sim | KV Savings | Mean SNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00% | 11.05% | 4.39%-21.04% | 4.59% | 1.000 | 0.00 MB | 7.44 dB |
| `uniform` | 49.96% | 35.02% | 26.66%-45.34% | 18.91% | 0.511 | 238.41 MB | -1.27 dB |
| `acoustic_salience` | 49.96% | 17.66% | 10.31%-28.19% | 8.28% | 0.830 | 238.41 MB | 0.12 dB |
| `energy_salience` | 49.96% | 18.04% | 10.63%-28.08% | 7.91% | 0.834 | 238.41 MB | 0.00 dB |
| `patch` | 74.93% | 99.84% | 99.52%-100.00% | 98.45% | 0.029 | 357.56 MB | -6.85 dB |

## Interpretation

Uniform frame dropping is cheap, but it damages intelligibility and voice identity. On this 24-clip slice, it increases WER from 11.05% to 35.02% and drops speaker similarity to 0.511.

The salience baselines keep the same token budget as uniform dropping but preserve ASR and speaker similarity much better:

- `acoustic_salience`: 17.66% WER
- `energy_salience`: 18.04% WER
- `uniform`: 35.02% WER

Energy salience slightly trails acoustic salience on WER but has the best compressed-speaker similarity in this run. That makes it the more interesting starting point for future VAD-aware work.

The `patch` result is intentionally bad. It confirms that naive arithmetic over discrete codec IDs is a failure mode, not a viable compression method.

## Failure Cases

The generated dashboard includes worst-case transcript rows and audio controls:

```text
modal-runs/encodec_librispeech_asr/dashboard.html
```

The main failure mode is not subtle: uniform frame dropping often turns words into plausible but wrong phrases. Patch averaging can collapse into empty or unrelated transcriptions. Salience methods still make word errors, especially on longer utterances, but they preserve enough local acoustic structure to stay far closer to the baseline.

## Launch Summary

Short version:

> I built AudioTokenLab, a benchmark for audio-token compression. On a 24-clip LibriSpeech EnCodec run, naive 2x frame dropping cut tokens by 50% but pushed WER to 35%. A simple salience policy kept the same 50% token reduction while cutting WER to about 18% and preserving speaker similarity much better.

Numbers to mention:

- 24 real speech clips
- 120 reconstructed samples
- Modal L4 run
- EnCodec 24 kHz tokens
- `faster-whisper` WER/CER
- SpeechBrain ECAPA speaker similarity
- 49.96% token reduction
- 35.02% WER for uniform dropping vs 17.66% WER for acoustic salience

## Current Limitations

- 24 clips is enough for a benchmark smoke result, not a publication-grade estimate.
- `faster-whisper` `tiny.en` is a convenient evaluator, not an oracle for speech quality.
- Speaker similarity is measured with one pretrained embedding model; subjective voice quality and prosody are outside the v1 metric scope.
- The energy baseline uses frame energy as a lightweight VAD proxy, not a trained speech activity detector.
- Token compression is evaluated through reconstruction and ASR, not through a downstream audio-language model yet.

## Next Research Steps

1. Run 100+ clips across LibriSpeech splits.
2. Add a stronger VAD backend and compare against frame-energy heuristics.
3. Evaluate semantic audio tokens or learned token selectors instead of purely heuristic frame selection.
4. Add subjective listening examples for the highest-impact failure cases.
5. Benchmark prefill/KV savings in an actual audio-token transformer once a target model is selected.
