# Listening Examples

These small WAV examples are committed under `experiments/results/listening_examples/`.

Selected clip: `5694-64025-0000`

WER can exceed 100% when insertions outnumber reference words.

| Strategy | WER | CER | Sample | Hypothesis |
| --- | ---: | ---: | --- | --- |
| `baseline` | 0.00% | 0.00% | [`wav`](listening_examples/5694-64025-0000__baseline.wav) | Shiloh. |
| `uniform` | 200.00% | 83.33% | [`wav`](listening_examples/5694-64025-0000__uniform.wav) | Shout out. |
| `acoustic_salience` | 0.00% | 0.00% | [`wav`](listening_examples/5694-64025-0000__acoustic_salience.wav) | Shiloh. |
| `energy_tuned_e4_t1_o2` | 0.00% | 0.00% | [`wav`](listening_examples/5694-64025-0000__energy_tuned_e4_t1_o2.wav) | Shiloh. |
| `patch` | 100.00% | 100.00% | [`wav`](listening_examples/5694-64025-0000__patch.wav) |  |
