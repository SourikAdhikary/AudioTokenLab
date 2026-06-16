from __future__ import annotations

import argparse
from pathlib import Path

from audiotokenlab.runner import run_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiotokenlab",
        description="Profile and compress audio-token workloads.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Run an AudioTokenLab profile job.")
    profile.add_argument("--config", required=True, help="Path to a JSON run config.")

    report = subparsers.add_parser("report", help="Print generated report paths.")
    report.add_argument("run_dir", help="Run directory produced by profile.")

    train_selector = subparsers.add_parser(
        "train-selector",
        help="Fit linear selector weights from an evaluated run directory.",
    )
    train_selector.add_argument("run_dir", help="Run directory with metrics/asr/speaker CSVs.")
    train_selector.add_argument(
        "--output",
        required=True,
        help="Path to write the trained selector JSON artifact.",
    )
    train_selector.add_argument(
        "--target-reduction",
        type=float,
        default=0.5,
        help="Target token reduction ratio for the training objective.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        rows = run_profile(args.config)
        print(f"wrote {len(rows)} metric rows")
        return 0

    if args.command == "report":
        run_dir = Path(args.run_dir)
        for name in (
            "manifest.json",
            "metrics.csv",
            "dashboard.html",
            "asr_metrics.csv",
            "asr_summary.json",
            "speaker_metrics.csv",
            "speaker_summary.json",
            "publication_summary.json",
            "summary_chart.svg",
            "listening_examples.md",
            "listening_study.csv",
            "listening_study.md",
            "serving_stack_report.json",
            "serving_stack_report.md",
        ):
            path = run_dir / name
            status = "ok" if path.exists() else "missing"
            print(f"{status}: {path}")
        return 0

    if args.command == "train-selector":
        from audiotokenlab.selector_training import train_selector_from_artifacts

        summary = train_selector_from_artifacts(
            Path(args.run_dir),
            Path(args.output),
            target_reduction=args.target_reduction,
        )
        strategy = summary["trained_strategy"]
        print(f"wrote trained selector: {args.output}")
        print(f"label: {strategy['label']}")
        print(f"weights: {strategy['weights']}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
