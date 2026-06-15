# Serving Stack Report

Stack: `reference_audio_token_transformer`

| Strategy | Mean Tokens | Token Reduction | KV Savings | Prefill Work Ratio | Decode KV Read Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `acoustic_salience` | 2328.0 | 49.95% | 217.75 MB | 0.251x | 49.94% |
| `baseline` | 4650.7 | 0.00% | 0.00 MB | 1.000x | 0.00% |
| `energy_salience` | 2328.0 | 49.95% | 217.75 MB | 0.251x | 49.94% |
| `linear_selector_v1` | 2328.0 | 49.95% | 217.75 MB | 0.251x | 49.94% |
| `patch` | 1165.3 | 74.95% | 326.75 MB | 0.063x | 74.94% |
| `uniform` | 2328.0 | 49.95% | 217.75 MB | 0.251x | 49.94% |
| `vad_salience` | 2328.0 | 49.95% | 217.75 MB | 0.251x | 49.94% |

## PyTorch Microbenchmark

- `acoustic_salience`: 1968 tokens, 3.43 ms prefill on cuda
- `baseline`: 3928 tokens, 11.81 ms prefill on cuda
- `energy_salience`: 1968 tokens, 4.48 ms prefill on cuda
- `linear_selector_v1`: 1968 tokens, 4.53 ms prefill on cuda
- `patch`: 984 tokens, 1.81 ms prefill on cuda
- `uniform`: 1968 tokens, 4.48 ms prefill on cuda
- `vad_salience`: 1968 tokens, 4.48 ms prefill on cuda
