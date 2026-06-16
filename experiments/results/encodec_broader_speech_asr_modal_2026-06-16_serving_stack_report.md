# Serving Stack Report

Stack: `reference_audio_token_transformer`

| Strategy | Mean Tokens | Token Reduction | KV Savings | Prefill Work Ratio | Decode KV Read Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `acoustic_salience` | 2489.2 | 49.95% | 232.97 MB | 0.250x | 49.96% |
| `baseline` | 4974.2 | 0.00% | 0.00 MB | 1.000x | 0.00% |
| `energy_salience` | 2489.2 | 49.95% | 232.97 MB | 0.250x | 49.96% |
| `linear_selector_v1` | 2489.2 | 49.95% | 232.97 MB | 0.250x | 49.96% |
| `patch` | 1246.5 | 74.92% | 349.47 MB | 0.063x | 74.94% |
| `uniform` | 2489.2 | 49.95% | 232.97 MB | 0.250x | 49.96% |
| `vad_salience` | 2489.2 | 49.95% | 232.97 MB | 0.250x | 49.96% |

## PyTorch Microbenchmark

- `acoustic_salience`: 2264 tokens, 4.08 ms prefill on cuda
- `baseline`: 4096 tokens, 10.56 ms prefill on cuda
- `energy_salience`: 2264 tokens, 4.15 ms prefill on cuda
- `linear_selector_v1`: 2264 tokens, 5.26 ms prefill on cuda
- `patch`: 1136 tokens, 2.01 ms prefill on cuda
- `uniform`: 2264 tokens, 5.53 ms prefill on cuda
- `vad_salience`: 2264 tokens, 5.46 ms prefill on cuda
