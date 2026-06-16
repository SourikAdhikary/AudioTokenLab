from __future__ import annotations

import json
import math
import importlib.util
import tempfile
import unittest
from pathlib import Path

from audiotokenlab.audio_io import write_wav
from audiotokenlab.asr_eval import (
    bootstrap_mean_ci,
    summarize_asr,
    write_asr_artifacts,
)
from audiotokenlab.compression import compress_tokens
from audiotokenlab.config import load_config
from audiotokenlab.corpora import merge_wav_manifests
from audiotokenlab.datasets import load_dataset
from audiotokenlab.librispeech import (
    parse_librispeech_transcripts,
    select_librispeech_flacs,
)
from audiotokenlab.models import TokenBundle
from audiotokenlab.profiling import estimate_kv_cache_mb, profile_clip
from audiotokenlab.publication import write_publication_artifacts
from audiotokenlab.reporting import summarize_by_strategy
from audiotokenlab.runner import run_profile
from audiotokenlab.listening_study import write_listening_study_artifacts
from audiotokenlab.selector_training import train_selector_from_artifacts
from audiotokenlab.serving import write_serving_stack_report
from audiotokenlab.speaker_eval import summarize_speaker, write_speaker_artifacts
from audiotokenlab.text_metrics import character_error_rate, word_error_rate
from audiotokenlab.tokenizers import build_tokenizer
from audiotokenlab.tokenizers.encodec_backend import _expanded_decode_tokens, _frame_energies


class PipelineTest(unittest.TestCase):
    def test_synthetic_dataset_and_dummy_tokenizer(self) -> None:
        clips = load_dataset(
            {
                "type": "synthetic",
                "count": 2,
                "sample_rate": 8000,
                "duration_seconds": 0.5,
            }
        )
        tokenizer = build_tokenizer({"name": "dummy", "frame_size": 80})
        bundle = tokenizer.encode(clips[0])
        decoded = tokenizer.decode(bundle)

        self.assertEqual(len(clips), 2)
        self.assertGreater(bundle.token_count, 0)
        self.assertEqual(len(decoded), len(clips[0].samples))

    def test_mulaw_tokenizer_round_trip(self) -> None:
        clip = load_dataset(
            {
                "type": "synthetic",
                "count": 1,
                "sample_rate": 8000,
                "duration_seconds": 0.25,
            }
        )[0]
        tokenizer = build_tokenizer(
            {"name": "mulaw", "quantization_channels": 256, "hop_size": 2}
        )
        bundle = tokenizer.encode(clip)
        decoded = tokenizer.decode(bundle)

        self.assertEqual(bundle.tokenizer, "mulaw")
        self.assertGreater(bundle.token_count, 0)
        self.assertEqual(len(decoded), len(clip.samples))

    def test_wav_manifest_loads_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "clip.wav"
            write_wav(audio_path, tuple([0.0] * 800), 8000)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip_id": "clip_a",
                                "path": str(audio_path),
                                "transcript": "hello audio tokens",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            clips = load_dataset({"type": "wav_manifest", "path": str(manifest_path)})

        self.assertEqual(clips[0].clip_id, "clip_a")
        self.assertEqual(clips[0].metadata["transcript"], "hello audio tokens")

    def test_merge_wav_manifests_keeps_dataset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "clip.wav"
            write_wav(audio_path, tuple([0.0] * 800), 8000)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip_id": "left_clip",
                                "path": str(audio_path),
                                "transcript": "left",
                                "source": "dataset_a",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip_id": "right_clip",
                                "path": str(audio_path),
                                "transcript": "right",
                                "source": "dataset_b",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            merged = merge_wav_manifests([left, right], root / "merged.json")
            clips = load_dataset({"type": "wav_manifest", "path": str(merged)})

        self.assertEqual([clip.clip_id for clip in clips], ["left_clip", "right_clip"])
        self.assertEqual(clips[1].metadata["source"], "dataset_b")

    def test_librispeech_transcript_parser(self) -> None:
        parsed = parse_librispeech_transcripts(
            "1272-128104-0000 A CHAPTER TITLE\n"
            "1272-128104-0001 THE QUICK BROWN FOX\n"
        )

        self.assertEqual(parsed["1272-128104-0000"], "A CHAPTER TITLE")
        self.assertEqual(parsed["1272-128104-0001"], "THE QUICK BROWN FOX")

    def test_librispeech_selection_spreads_speakers_and_chapters(self) -> None:
        names = [
            "LibriSpeech/dev-clean/111/100/111-100-0000.flac",
            "LibriSpeech/dev-clean/111/100/111-100-0001.flac",
            "LibriSpeech/dev-clean/222/200/222-200-0000.flac",
            "LibriSpeech/dev-clean/222/200/222-200-0001.flac",
            "LibriSpeech/dev-clean/333/300/333-300-0000.flac",
        ]
        transcripts = {
            "111-100-0000": "A",
            "111-100-0001": "B",
            "222-200-0000": "C",
            "222-200-0001": "D",
            "333-300-0000": "E",
        }

        selected = select_librispeech_flacs(names, transcripts, max_clips=4)

        self.assertEqual(
            selected,
            [
                "LibriSpeech/dev-clean/111/100/111-100-0000.flac",
                "LibriSpeech/dev-clean/222/200/222-200-0000.flac",
                "LibriSpeech/dev-clean/333/300/333-300-0000.flac",
                "LibriSpeech/dev-clean/111/100/111-100-0001.flac",
            ],
        )

    def test_compression_reduces_tokens(self) -> None:
        clip = load_dataset({"type": "synthetic", "count": 1})[0]
        tokenizer = build_tokenizer({"name": "dummy", "frame_size": 160})
        bundle = tokenizer.encode(clip)
        compressed = compress_tokens(bundle, {"name": "uniform", "factor": 2})

        self.assertLess(compressed.token_count, bundle.token_count)

    def test_silence_aware_compression_reduces_quiet_tokens(self) -> None:
        clip = load_dataset(
            {
                "type": "synthetic_quiet",
                "count": 1,
                "sample_rate": 8000,
                "duration_seconds": 1.0,
                "speech_seconds": 0.15,
                "quiet_seconds": 0.35,
            }
        )[0]
        tokenizer = build_tokenizer({"name": "dummy", "frame_size": 80})
        bundle = tokenizer.encode(clip)
        compressed = compress_tokens(
            bundle,
            {"name": "silence_aware", "factor": 3, "threshold": 4},
        )

        self.assertLess(compressed.token_count, bundle.token_count)

    def test_silence_aware_supports_center_coded_quiet_tokens(self) -> None:
        clip = load_dataset(
            {
                "type": "synthetic_quiet",
                "count": 1,
                "sample_rate": 8000,
                "duration_seconds": 1.0,
                "speech_seconds": 0.15,
                "quiet_seconds": 0.35,
            }
        )[0]
        tokenizer = build_tokenizer(
            {"name": "mulaw", "quantization_channels": 256, "hop_size": 4}
        )
        bundle = tokenizer.encode(clip)
        compressed = compress_tokens(
            bundle,
            {
                "name": "silence_aware",
                "factor": 3,
                "threshold_low": 124,
                "threshold_high": 132,
            },
        )

        self.assertLess(compressed.token_count, bundle.token_count)

    def test_frame_major_uniform_compression_preserves_codebook_groups(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 2, 20, 3, 30, 4, 40),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.08,
            metadata={"token_layout": "frame_major", "frame_count": 4},
        )
        compressed = compress_tokens(bundle, {"name": "uniform", "factor": 2})

        self.assertEqual(compressed.tokens, (1, 10, 3, 30))
        self.assertEqual(compressed.metadata["frame_count"], 2)

    def test_frame_major_patch_compression_preserves_codebook_groups(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 3, 30, 5, 50, 7, 70),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.08,
            metadata={"token_layout": "frame_major", "frame_count": 4},
        )
        compressed = compress_tokens(bundle, {"name": "patch", "patch_size": 2})

        self.assertEqual(compressed.tokens, (2, 20, 6, 60))
        self.assertEqual(compressed.metadata["frame_count"], 2)

    def test_acoustic_salience_keeps_high_transition_frames(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 1, 10, 20, 200, 21, 201, 22, 202, 80, 800),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.12,
            metadata={"token_layout": "frame_major", "frame_count": 6},
        )
        compressed = compress_tokens(bundle, {"name": "acoustic_salience", "factor": 3})

        self.assertEqual(compressed.tokens, (20, 200, 80, 800))
        self.assertEqual(compressed.metadata["frame_count"], 2)
        self.assertEqual(compressed.metadata["decode_repeat_counts"], [3, 3])
        self.assertEqual(compressed.metadata["decode_frame_count"], 6)

    def test_decode_repeat_counts_expand_frame_major_tokens(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(7, 70, 8, 80),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.08,
            metadata={
                "token_layout": "frame_major",
                "frame_count": 2,
                "decode_repeat_counts": [2, 1],
            },
        )

        self.assertEqual(_expanded_decode_tokens(bundle), (7, 70, 7, 70, 8, 80))

    def test_energy_salience_uses_frame_energy_metadata(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.12,
            metadata={
                "token_layout": "frame_major",
                "frame_count": 6,
                "frame_energies": [0.01, 0.9, 0.02, 0.01, 0.03, 0.8],
            },
        )
        compressed = compress_tokens(
            bundle,
            {
                "name": "energy_salience",
                "factor": 3,
                "energy_weight": 10.0,
                "transition_weight": 0.0,
                "onset_weight": 0.0,
            },
        )

        self.assertEqual(compressed.tokens, (2, 11, 6, 15))
        self.assertEqual(compressed.metadata["decode_repeat_counts"], [3, 3])

    def test_vad_salience_prefers_speech_activity_frames(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.12,
            metadata={
                "token_layout": "frame_major",
                "frame_count": 6,
                "frame_energies": [0.01, 0.02, 0.9, 0.8, 0.03, 0.02],
            },
        )
        compressed = compress_tokens(
            bundle,
            {
                "name": "vad_salience",
                "factor": 3,
                "absolute_threshold": 0.2,
                "min_speech_frames": 1,
                "hangover_frames": 0,
                "transition_weight": 0.0,
                "onset_weight": 1.0,
            },
        )

        self.assertEqual(compressed.tokens, (3, 12, 4, 13))
        self.assertEqual(compressed.metadata["decode_repeat_counts"], [3, 3])
        self.assertGreater(compressed.metadata["speech_activity_ratio"], 0.0)

    def test_learned_selector_accepts_linear_weights(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 2, 11, 3, 12, 4, 13),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.08,
            metadata={
                "token_layout": "frame_major",
                "frame_count": 4,
                "frame_energies": [0.1, 0.7, 0.2, 0.8],
            },
        )
        compressed = compress_tokens(
            bundle,
            {
                "name": "learned_selector",
                "factor": 2,
                "weights": {
                    "energy": 10.0,
                    "transition": 0.0,
                    "onset": 0.0,
                    "speech_activity": 0.0,
                    "center": 0.0,
                },
            },
        )

        self.assertEqual(compressed.tokens, (2, 11, 4, 13))
        self.assertEqual(compressed.metadata["selector_type"], "linear_frame_selector")

    def test_learned_selector_loads_weights_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_path = root / "selector.json"
            weights_path.write_text(
                json.dumps(
                    {
                        "trained_strategy": {
                            "weights": {
                                "energy": 10.0,
                                "transition": 0.0,
                                "onset": 0.0,
                                "speech_activity": 0.0,
                                "center": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            bundle = TokenBundle(
                clip_id="rvq",
                tokenizer="test",
                tokens=(1, 10, 2, 11, 3, 12, 4, 13),
                frame_rate=50.0,
                codebook_count=2,
                sample_rate=24000,
                duration_seconds=0.08,
                metadata={
                    "token_layout": "frame_major",
                    "frame_count": 4,
                    "frame_energies": [0.1, 0.7, 0.2, 0.8],
                },
            )

            compressed = compress_tokens(
                bundle,
                {
                    "name": "learned_selector",
                    "factor": 2,
                    "weights_path": str(weights_path),
                },
            )

        self.assertEqual(compressed.tokens, (2, 11, 4, 13))

    def test_strategy_label_is_used_for_metric_identity(self) -> None:
        bundle = TokenBundle(
            clip_id="rvq",
            tokenizer="test",
            tokens=(1, 10, 2, 11, 3, 12, 4, 13),
            frame_rate=50.0,
            codebook_count=2,
            sample_rate=24000,
            duration_seconds=0.08,
            metadata={
                "token_layout": "frame_major",
                "frame_count": 4,
                "frame_energies": [0.01, 0.9, 0.02, 0.8],
            },
        )
        compressed = compress_tokens(
            bundle,
            {
                "name": "energy_salience",
                "label": "energy_tuned_test",
                "factor": 2,
            },
        )

        self.assertEqual(compressed.metadata["compression_strategy"], "energy_tuned_test")

    def test_frame_energies_match_requested_frame_count(self) -> None:
        energies = _frame_energies((0.0, 1.0, 0.0, 2.0), frame_count=2)

        self.assertEqual(len(energies), 2)
        self.assertAlmostEqual(energies[0], 0.5)
        self.assertAlmostEqual(energies[1], 2.0)

    def test_encodec_backend_reports_missing_optional_dependency(self) -> None:
        if importlib.util.find_spec("encodec") is not None:
            self.skipTest("encodec is installed in this environment")
        with self.assertRaisesRegex(ImportError, "optional dependencies"):
            build_tokenizer({"name": "encodec"})

    def test_profile_clip_emits_metric_rows(self) -> None:
        clip = load_dataset({"type": "synthetic", "count": 1})[0]
        tokenizer = build_tokenizer({"name": "dummy"})
        rows = profile_clip(
            clip,
            tokenizer,
            [{"name": "baseline"}, {"name": "patch", "patch_size": 4}],
            {"layers": 2, "hidden_size": 16, "bytes_per_element": 2},
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].strategy, "baseline")
        self.assertGreaterEqual(rows[1].token_reduction_ratio, 0.0)
        self.assertTrue(math.isfinite(rows[0].reconstruction_snr_db))

    def test_strategy_summary_groups_rows(self) -> None:
        clip = load_dataset({"type": "synthetic", "count": 1})[0]
        tokenizer = build_tokenizer({"name": "dummy"})
        rows = profile_clip(
            clip,
            tokenizer,
            [{"name": "baseline"}, {"name": "uniform", "factor": 2}],
            {"layers": 2, "hidden_size": 16, "bytes_per_element": 2},
        )
        summary = summarize_by_strategy(rows)

        self.assertEqual(sorted(summary), ["baseline", "uniform"])
        self.assertEqual(summary["baseline"]["row_count"], 1)
        self.assertGreater(summary["uniform"]["mean_token_reduction_ratio"], 0.0)

    def test_kv_cache_estimate(self) -> None:
        mb = estimate_kv_cache_mb(
            token_count=10,
            kv_cache={"layers": 2, "hidden_size": 4, "bytes_per_element": 2},
        )
        self.assertAlmostEqual(mb, 10 * 2 * 4 * 2 * 2 / (1024 * 1024))

    def test_text_error_rates(self) -> None:
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)
        self.assertEqual(word_error_rate("hello world", "hello"), 0.5)
        self.assertAlmostEqual(character_error_rate("abc", "adc"), 1 / 3)

    def test_asr_summary_groups_strategies(self) -> None:
        summary = summarize_asr(
            [
                {"strategy": "baseline", "wer": 0.0, "cer": 0.0},
                {"strategy": "patch", "wer": 0.5, "cer": 0.25},
            ]
        )

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["strategy_summary"]["baseline"]["mean_wer"], 0.0)
        self.assertEqual(summary["strategy_summary"]["patch"]["mean_cer"], 0.25)
        self.assertEqual(
            summary["strategy_summary"]["baseline"]["wer_ci95"],
            {"low": 0.0, "high": 0.0},
        )

    def test_bootstrap_mean_ci_is_deterministic(self) -> None:
        rows = [{"wer": 0.0}, {"wer": 0.5}, {"wer": 1.0}]

        first = bootstrap_mean_ci(rows, "wer", iterations=100)
        second = bootstrap_mean_ci(rows, "wer", iterations=100)

        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["high"])

    def test_asr_artifacts_refresh_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "asr_test",
                        "summary": {
                            "row_count": 2,
                            "clip_count": 1,
                            "mean_token_reduction_ratio": 0.25,
                            "mean_kv_cache_savings_mb": 3.0,
                        },
                        "strategy_summary": {
                            "baseline": {
                                "mean_token_reduction_ratio": 0.0,
                                "mean_kv_cache_savings_mb": 0.0,
                                "mean_reconstruction_snr_db": 9.0,
                                "mean_real_time_factor": 0.01,
                            },
                            "uniform": {
                                "mean_token_reduction_ratio": 0.5,
                                "mean_kv_cache_savings_mb": 6.0,
                                "mean_reconstruction_snr_db": -1.0,
                                "mean_real_time_factor": 0.02,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "metrics.csv").write_text(
                "clip_id,strategy,original_tokens,compressed_tokens,"
                "token_reduction_ratio,estimated_kv_cache_savings_mb,"
                "reconstruction_snr_db,real_time_factor\n"
                "clip_a,baseline,10,10,0,0,9,0.01\n"
                "clip_a,uniform,10,5,0.5,6,-1,0.02\n",
                encoding="utf-8",
            )

            write_asr_artifacts(
                root,
                [
                    {
                        "clip_id": "clip_a",
                        "strategy": "baseline",
                        "sample_path": "/tmp/clip_a__baseline.wav",
                        "reference_text": "hello audio",
                        "hypothesis_text": "hello audio",
                        "wer": 0.0,
                        "cer": 0.0,
                    },
                    {
                        "clip_id": "clip_a",
                        "strategy": "uniform",
                        "sample_path": "/tmp/clip_a__uniform.wav",
                        "reference_text": "hello audio",
                        "hypothesis_text": "hello",
                        "wer": 0.5,
                        "cer": 0.45,
                    },
                ],
            )

            dashboard = (root / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn("AudioTokenLab ASR Benchmark", dashboard)
        self.assertIn("Token Budget vs ASR", dashboard)
        self.assertIn("Worst ASR Cases", dashboard)
        self.assertIn("WER 95% CI", dashboard)
        self.assertIn("50.00%", dashboard)

    def test_speaker_summary_groups_strategies(self) -> None:
        summary = summarize_speaker(
            [
                {"strategy": "baseline", "speaker_similarity": 1.0},
                {"strategy": "uniform", "speaker_similarity": 0.5},
            ]
        )

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(
            summary["strategy_summary"]["baseline"]["mean_speaker_similarity"],
            1.0,
        )
        self.assertEqual(
            summary["strategy_summary"]["uniform"]["mean_speaker_similarity"],
            0.5,
        )

    def test_speaker_artifacts_refresh_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "speaker_test",
                        "summary": {
                            "row_count": 2,
                            "clip_count": 1,
                            "mean_token_reduction_ratio": 0.25,
                            "mean_kv_cache_savings_mb": 3.0,
                        },
                        "strategy_summary": {
                            "baseline": {
                                "mean_token_reduction_ratio": 0.0,
                                "mean_kv_cache_savings_mb": 0.0,
                                "mean_reconstruction_snr_db": 9.0,
                                "mean_real_time_factor": 0.01,
                            },
                            "uniform": {
                                "mean_token_reduction_ratio": 0.5,
                                "mean_kv_cache_savings_mb": 6.0,
                                "mean_reconstruction_snr_db": -1.0,
                                "mean_real_time_factor": 0.02,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "metrics.csv").write_text(
                "clip_id,strategy,original_tokens,compressed_tokens,"
                "token_reduction_ratio,estimated_kv_cache_savings_mb,"
                "reconstruction_snr_db,real_time_factor\n"
                "clip_a,baseline,10,10,0,0,9,0.01\n"
                "clip_a,uniform,10,5,0.5,6,-1,0.02\n",
                encoding="utf-8",
            )
            (root / "asr_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_text,hypothesis_text,wer,cer\n"
                "clip_a,baseline,/tmp/clip_a__baseline.wav,hello,hello,0,0\n"
                "clip_a,uniform,/tmp/clip_a__uniform.wav,hello,helo,0.5,0.25\n",
                encoding="utf-8",
            )

            write_speaker_artifacts(
                root,
                [
                    {
                        "clip_id": "clip_a",
                        "strategy": "baseline",
                        "sample_path": "/tmp/clip_a__baseline.wav",
                        "reference_strategy": "baseline",
                        "speaker_similarity": 1.0,
                        "model_source": "test",
                    },
                    {
                        "clip_id": "clip_a",
                        "strategy": "uniform",
                        "sample_path": "/tmp/clip_a__uniform.wav",
                        "reference_strategy": "baseline",
                        "speaker_similarity": 0.75,
                        "model_source": "test",
                    },
                ],
            )

            dashboard = (root / "dashboard.html").read_text(encoding="utf-8")
            speaker_summary = json.loads(
                (root / "speaker_summary.json").read_text(encoding="utf-8")
            )

        self.assertIn("Speaker Sim", dashboard)
        self.assertIn("0.750", dashboard)
        self.assertEqual(speaker_summary["row_count"], 2)

    def test_publication_artifacts_write_chart_and_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics.csv").write_text(
                "clip_id,strategy,token_reduction_ratio,estimated_kv_cache_savings_mb,"
                "reconstruction_snr_db\n"
                "clip_a,baseline,0,0,9\n"
                "clip_a,uniform,0.5,6,-1\n"
                "clip_a,energy_tuned,0.5,6,0\n",
                encoding="utf-8",
            )
            (root / "asr_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_text,hypothesis_text,wer,cer\n"
                "clip_a,baseline,/tmp/clip_a__baseline.wav,hello,hello,0,0\n"
                "clip_a,uniform,/tmp/clip_a__uniform.wav,hello,helo,0.5,0.25\n"
                "clip_a,energy_tuned,/tmp/clip_a__energy_tuned.wav,hello,hello,0,0\n",
                encoding="utf-8",
            )
            (root / "speaker_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_strategy,speaker_similarity,model_source\n"
                "clip_a,baseline,/tmp/clip_a__baseline.wav,baseline,1,test\n"
                "clip_a,uniform,/tmp/clip_a__uniform.wav,baseline,0.5,test\n"
                "clip_a,energy_tuned,/tmp/clip_a__energy_tuned.wav,baseline,0.9,test\n",
                encoding="utf-8",
            )

            summary = write_publication_artifacts(root)
            chart_exists = (root / "summary_chart.svg").exists()
            examples_exists = (root / "listening_examples.md").exists()
            listening_study_exists = (root / "listening_study.csv").exists()
            serving_report_exists = (root / "serving_stack_report.json").exists()
            examples_text = (root / "listening_examples.md").read_text(
                encoding="utf-8",
            )

        self.assertEqual(summary["best_energy_strategy"], "energy_tuned")
        self.assertTrue(chart_exists)
        self.assertTrue(examples_exists)
        self.assertTrue(listening_study_exists)
        self.assertTrue(serving_report_exists)
        self.assertIn("energy_tuned", examples_text)

    def test_listening_study_artifacts_write_rating_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asr_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_text,hypothesis_text,wer,cer\n"
                "clip_a,baseline,/tmp/clip_a__baseline.wav,hello,hello,0,0\n"
                "clip_a,uniform,/tmp/clip_a__uniform.wav,hello,helo,0.5,0.25\n",
                encoding="utf-8",
            )
            (root / "speaker_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_strategy,speaker_similarity,model_source\n"
                "clip_a,baseline,/tmp/clip_a__baseline.wav,baseline,1,test\n"
                "clip_a,uniform,/tmp/clip_a__uniform.wav,baseline,0.5,test\n",
                encoding="utf-8",
            )

            summary = write_listening_study_artifacts(root)
            rating_sheet = (root / "listening_study.csv").read_text(encoding="utf-8")

        self.assertEqual(summary["item_count"], 1)
        self.assertIn("mos_1_5", rating_sheet)
        self.assertIn("atl_0001", rating_sheet)

    def test_serving_stack_report_estimates_transformer_savings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics.csv").write_text(
                "clip_id,strategy,original_tokens,compressed_tokens,"
                "token_reduction_ratio,estimated_kv_cache_savings_mb\n"
                "clip_a,baseline,100,100,0,0\n"
                "clip_a,uniform,100,50,0.5,6\n",
                encoding="utf-8",
            )

            report = write_serving_stack_report(root)

        self.assertEqual(report["strategy_summary"]["uniform"]["prefill_attention_work_ratio"], 0.25)
        self.assertEqual(report["strategy_summary"]["uniform"]["decode_kv_read_reduction_ratio"], 0.5)

    def test_train_selector_from_artifacts_writes_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics.csv").write_text(
                "clip_id,strategy,token_reduction_ratio,reconstruction_snr_db\n"
                "clip_a,acoustic_salience,0.5,0\n"
                "clip_a,energy_salience,0.5,0\n"
                "clip_a,vad_salience,0.5,0\n"
                "clip_a,linear_selector_v1,0.5,0\n",
                encoding="utf-8",
            )
            (root / "asr_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_text,hypothesis_text,wer,cer\n"
                "clip_a,acoustic_salience,/tmp/a.wav,hello,hello,0.3,0.2\n"
                "clip_a,energy_salience,/tmp/e.wav,hello,hello,0.1,0.1\n"
                "clip_a,vad_salience,/tmp/v.wav,hello,hello,0.2,0.1\n"
                "clip_a,linear_selector_v1,/tmp/l.wav,hello,hello,0.25,0.1\n",
                encoding="utf-8",
            )
            (root / "speaker_metrics.csv").write_text(
                "clip_id,strategy,sample_path,reference_strategy,speaker_similarity,model_source\n"
                "clip_a,acoustic_salience,/tmp/a.wav,baseline,0.7,test\n"
                "clip_a,energy_salience,/tmp/e.wav,baseline,0.9,test\n"
                "clip_a,vad_salience,/tmp/v.wav,baseline,0.8,test\n"
                "clip_a,linear_selector_v1,/tmp/l.wav,baseline,0.75,test\n",
                encoding="utf-8",
            )
            output_path = root / "trained_selector.json"

            summary = train_selector_from_artifacts(root, output_path)
            output_exists = output_path.exists()

        self.assertTrue(output_exists)
        self.assertEqual(summary["trained_strategy"]["label"], "trained_selector_v1")
        self.assertGreater(summary["strategy_credit"]["energy_salience"], 0.0)
        self.assertIn("weights", summary["trained_strategy"])

    def test_end_to_end_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            output_dir = root / "run"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "test",
                        "output_dir": str(output_dir),
                        "dataset": {
                            "type": "synthetic",
                            "count": 2,
                            "sample_rate": 8000,
                            "duration_seconds": 0.25,
                        },
                        "tokenizer": {"name": "dummy", "frame_size": 80},
                        "strategies": [
                            {"name": "baseline"},
                            {"name": "uniform", "factor": 2},
                        ],
                        "kv_cache": {
                            "layers": 2,
                            "hidden_size": 16,
                            "bytes_per_element": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = run_profile(str(config_path))
            manifest = load_config(config_path)

            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest.run_id, "test")
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "metrics.csv").exists())
            self.assertTrue((output_dir / "dashboard.html").exists())
            self.assertTrue((output_dir / "samples" / "synthetic_000__baseline.wav").exists())


if __name__ == "__main__":
    unittest.main()
