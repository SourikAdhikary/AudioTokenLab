from __future__ import annotations

import json
import math
import importlib.util
import tempfile
import unittest
from pathlib import Path

from audiotokenlab.audio_io import write_wav
from audiotokenlab.asr_eval import summarize_asr
from audiotokenlab.compression import compress_tokens
from audiotokenlab.config import load_config
from audiotokenlab.datasets import load_dataset
from audiotokenlab.librispeech import parse_librispeech_transcripts
from audiotokenlab.models import TokenBundle
from audiotokenlab.profiling import estimate_kv_cache_mb, profile_clip
from audiotokenlab.reporting import summarize_by_strategy
from audiotokenlab.runner import run_profile
from audiotokenlab.text_metrics import character_error_rate, word_error_rate
from audiotokenlab.tokenizers import build_tokenizer


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

    def test_librispeech_transcript_parser(self) -> None:
        parsed = parse_librispeech_transcripts(
            "1272-128104-0000 A CHAPTER TITLE\n"
            "1272-128104-0001 THE QUICK BROWN FOX\n"
        )

        self.assertEqual(parsed["1272-128104-0000"], "A CHAPTER TITLE")
        self.assertEqual(parsed["1272-128104-0001"], "THE QUICK BROWN FOX")

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
