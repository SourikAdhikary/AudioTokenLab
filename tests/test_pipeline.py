from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audiotokenlab.compression import compress_tokens
from audiotokenlab.config import load_config
from audiotokenlab.datasets import load_dataset
from audiotokenlab.profiling import estimate_kv_cache_mb, profile_clip
from audiotokenlab.runner import run_profile
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

    def test_compression_reduces_tokens(self) -> None:
        clip = load_dataset({"type": "synthetic", "count": 1})[0]
        tokenizer = build_tokenizer({"name": "dummy", "frame_size": 160})
        bundle = tokenizer.encode(clip)
        compressed = compress_tokens(bundle, {"name": "uniform", "factor": 2})

        self.assertLess(compressed.token_count, bundle.token_count)

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

    def test_kv_cache_estimate(self) -> None:
        mb = estimate_kv_cache_mb(
            token_count=10,
            kv_cache={"layers": 2, "hidden_size": 4, "bytes_per_element": 2},
        )
        self.assertAlmostEqual(mb, 10 * 2 * 4 * 2 * 2 / (1024 * 1024))

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


if __name__ == "__main__":
    unittest.main()

