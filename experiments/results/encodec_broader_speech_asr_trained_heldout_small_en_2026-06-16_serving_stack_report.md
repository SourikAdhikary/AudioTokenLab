# Serving Stack Report

Stack: `reference_audio_token_transformer`

| Strategy | Mean Tokens | Token Reduction | KV Savings | Prefill Work Ratio | Decode KV Read Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `acoustic_salience` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |
| `baseline` | 4873.3 | 0.00% | 0.00 MB | 1.000x | 0.00% |
| `energy_salience` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |
| `linear_selector_v1` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |
| `patch` | 1220.8 | 74.94% | 342.43 MB | 0.063x | 74.95% |
| `trained_selector_v1` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |
| `uniform` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |
| `vad_salience` | 2438.7 | 49.95% | 228.25 MB | 0.250x | 49.96% |

## PyTorch Microbenchmark

- `acoustic_salience`: 2080 tokens, 3.78 ms prefill on cuda
- `baseline`: 4096 tokens, 12.21 ms prefill on cuda
- `energy_salience`: 2080 tokens, 4.61 ms prefill on cuda
- `linear_selector_v1`: 2080 tokens, 4.74 ms prefill on cuda
- `patch`: 1040 tokens, 1.82 ms prefill on cuda
- `trained_selector_v1`: 2080 tokens, 4.74 ms prefill on cuda
- `uniform`: 2080 tokens, 4.85 ms prefill on cuda
- `vad_salience`: 2080 tokens, 4.91 ms prefill on cuda
