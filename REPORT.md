# AudioTokenLab Benchmark Report

## Summary

AudioTokenLab benchmarks how much EnCodec audio-token streams can be compressed before downstream speech utility breaks.

The current benchmark evaluates EnCodec 24 kHz reconstructions with `faster-whisper` on a small but real LibriSpeech `dev-clean` slice. The main result: simple salience-based sparse-frame retention gives the same roughly 50% token reduction as uniform frame dropping, but preserves ASR much better.

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
- Modal run: https://modal.com/apps/sourikadhikary/main/ap-47G8kWlbXHVg4RMx48rLFC

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

| Strategy | Token Reduction | Mean WER | Mean CER | KV Savings | Mean SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.00% | 11.05% | 4.59% | 0.00 MB | 7.44 dB |
| `uniform` | 49.96% | 36.57% | 19.52% | 238.41 MB | -1.27 dB |
| `acoustic_salience` | 49.96% | 17.66% | 8.28% | 238.41 MB | 0.12 dB |
| `energy_salience` | 49.96% | 18.04% | 7.91% | 238.41 MB | 0.00 dB |
| `patch` | 74.93% | 99.84% | 98.45% | 357.56 MB | -6.85 dB |

## Interpretation

Uniform frame dropping is cheap, but it damages intelligibility. On this 24-clip slice, it increases WER from 11.05% to 36.57%.

The salience baselines keep the same token budget as uniform dropping but preserve ASR much better:

- `acoustic_salience`: 17.66% WER
- `energy_salience`: 18.04% WER
- `uniform`: 36.57% WER

That is the first useful optimization target for the project: keep the 50% token reduction while pushing WER closer to the EnCodec baseline.

The `patch` result is intentionally bad. It confirms that naive arithmetic over discrete codec IDs is a failure mode, not a viable compression method.

## Current Limitations

- 24 clips is enough for a benchmark smoke result, not a publication-grade estimate.
- `faster-whisper` `tiny.en` is a convenient evaluator, not an oracle for speech quality.
- Speaker similarity, prosody, and subjective audio quality are not measured yet.
- The energy/VAD baseline uses frame energy as a lightweight proxy, not a trained speech activity detector.
- Token compression is evaluated through reconstruction and ASR, not through a downstream audio-language model yet.

## Next Research Steps

1. Run 100+ clips across LibriSpeech splits and report confidence intervals.
2. Add speaker embedding similarity.
3. Add a stronger VAD backend and compare against frame-energy heuristics.
4. Evaluate semantic audio tokens or learned token selectors instead of purely heuristic frame selection.
5. Benchmark prefill/KV savings in an actual audio-token transformer once a target model is selected.
