from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import modal


APP_DIR = Path("/root/audiotokenlab")
DEFAULT_CONFIG = "experiments/configs/encodec_demo.json"
DEFAULT_LOCAL_OUT = Path("modal-runs")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "espeak-ng", "ffmpeg")
    .pip_install("torch>=2.0", "encodec>=0.1.1", "faster-whisper>=1.0.0")
    .env({"PYTHONPATH": str(APP_DIR / "src")})
    .workdir(str(APP_DIR))
    .add_local_dir("src", str(APP_DIR / "src"))
    .add_local_dir("experiments", str(APP_DIR / "experiments"))
)

app = modal.App("audiotokenlab", image=image)


@app.function(gpu="L4", timeout=20 * 60, memory=8192)
def run_encodec_profile(config_path: str = DEFAULT_CONFIG) -> dict:
    from audiotokenlab.runner import run_profile

    rows = run_profile(config_path)
    config = _load_json_config(APP_DIR / config_path)
    output_dir = _resolve_output_dir(APP_DIR / config_path, config.get("output_dir"))
    archive = _zip_directory(output_dir)
    return {
        "run_id": config.get("run_id", "encodec_demo"),
        "row_count": len(rows),
        "output_dir": str(output_dir),
        "archive_name": f"{config.get('run_id', 'encodec_demo')}.zip",
        "archive_bytes": archive,
    }


@app.function(gpu="L4", timeout=25 * 60, memory=8192)
def run_speech_asr_profile() -> dict:
    from audiotokenlab.asr_eval import (
        transcribe_samples_with_faster_whisper,
        write_asr_artifacts,
    )
    from audiotokenlab.runner import run_profile

    work_dir = Path("/tmp/audiotokenlab_speech")
    audio_dir = work_dir / "audio"
    output_dir = work_dir / "runs" / "encodec_speech_asr"
    audio_dir.mkdir(parents=True, exist_ok=True)

    transcripts = {
        "speech_000": "audio token compression should preserve the words",
        "speech_001": "the benchmark measures latency memory and quality",
    }
    manifest_clips = []
    for clip_id, text in transcripts.items():
        raw_path = audio_dir / f"{clip_id}_raw.wav"
        wav_path = audio_dir / f"{clip_id}.wav"
        subprocess.run(["espeak-ng", "-w", str(raw_path), text], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-ar",
                "24000",
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
        )
        manifest_clips.append(
            {
                "clip_id": clip_id,
                "path": str(wav_path),
                "transcript": text,
            }
        )

    manifest_path = work_dir / "speech_manifest.json"
    manifest_path.write_text(
        json.dumps({"root": str(audio_dir), "clips": manifest_clips}, indent=2),
        encoding="utf-8",
    )
    config_path = work_dir / "speech_asr_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "encodec_speech_asr",
                "output_dir": str(output_dir),
                "dataset": {"type": "wav_manifest", "path": str(manifest_path)},
                "tokenizer": {
                    "name": "encodec",
                    "model_name": "encodec_24khz",
                    "bandwidth": 6.0,
                    "device": "cuda",
                },
                "strategies": [
                    {"name": "baseline"},
                    {"name": "uniform", "factor": 2},
                    {"name": "acoustic_salience", "factor": 2},
                    {"name": "patch", "patch_size": 4},
                ],
                "kv_cache": {
                    "layers": 24,
                    "hidden_size": 1024,
                    "bytes_per_element": 2,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = run_profile(str(config_path))
    asr_rows = transcribe_samples_with_faster_whisper(
        output_dir / "samples",
        transcripts,
        model_name="tiny.en",
        device="cpu",
        compute_type="int8",
    )
    write_asr_artifacts(output_dir, asr_rows)
    archive = _zip_directory(output_dir)
    return {
        "run_id": "encodec_speech_asr",
        "row_count": len(rows),
        "asr_row_count": len(asr_rows),
        "output_dir": str(output_dir),
        "archive_name": "encodec_speech_asr.zip",
        "archive_bytes": archive,
    }


@app.function(gpu="L4", timeout=35 * 60, memory=8192)
def run_librispeech_asr_profile(max_clips: int = 4) -> dict:
    from audiotokenlab.asr_eval import (
        transcribe_samples_with_faster_whisper,
        write_asr_artifacts,
    )
    from audiotokenlab.datasets import load_dataset
    from audiotokenlab.librispeech import prepare_librispeech_slice
    from audiotokenlab.runner import run_profile

    work_dir = Path("/tmp/audiotokenlab_librispeech")
    dataset_dir = work_dir / "dataset"
    output_dir = work_dir / "runs" / "encodec_librispeech_asr"
    manifest_path = prepare_librispeech_slice(
        dataset_dir,
        max_clips=max_clips,
        sample_rate=24000,
    )
    clips = load_dataset({"type": "wav_manifest", "path": str(manifest_path)})
    references = {
        clip.clip_id: str(clip.metadata.get("transcript", ""))
        for clip in clips
    }
    config_path = work_dir / "librispeech_asr_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "encodec_librispeech_asr",
                "output_dir": str(output_dir),
                "dataset": {"type": "wav_manifest", "path": str(manifest_path)},
                "tokenizer": {
                    "name": "encodec",
                    "model_name": "encodec_24khz",
                    "bandwidth": 6.0,
                    "device": "cuda",
                },
                "strategies": [
                    {"name": "baseline"},
                    {"name": "uniform", "factor": 2},
                    {"name": "acoustic_salience", "factor": 2},
                    {"name": "patch", "patch_size": 4},
                ],
                "kv_cache": {
                    "layers": 24,
                    "hidden_size": 1024,
                    "bytes_per_element": 2,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = run_profile(str(config_path))
    asr_rows = transcribe_samples_with_faster_whisper(
        output_dir / "samples",
        references,
        model_name="tiny.en",
        device="cpu",
        compute_type="int8",
    )
    write_asr_artifacts(output_dir, asr_rows)
    archive = _zip_directory(output_dir)
    return {
        "run_id": "encodec_librispeech_asr",
        "row_count": len(rows),
        "asr_row_count": len(asr_rows),
        "clip_count": len(clips),
        "output_dir": str(output_dir),
        "archive_name": "encodec_librispeech_asr.zip",
        "archive_bytes": archive,
    }


@app.local_entrypoint()
def main(
    config_path: str = DEFAULT_CONFIG,
    local_out: str = str(DEFAULT_LOCAL_OUT),
    extract: bool = True,
    speech_asr: bool = False,
    librispeech_asr: bool = False,
    max_clips: int = 4,
) -> None:
    if librispeech_asr:
        result = run_librispeech_asr_profile.remote(max_clips)
    elif speech_asr:
        result = run_speech_asr_profile.remote()
    else:
        result = run_encodec_profile.remote(config_path)
    target_root = Path(local_out)
    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = target_root / str(result["archive_name"])
    archive_path.write_bytes(result["archive_bytes"])
    print(f"wrote archive: {archive_path}")
    print(f"remote rows: {result['row_count']}")
    if "asr_row_count" in result:
        print(f"asr rows: {result['asr_row_count']}")
    if "clip_count" in result:
        print(f"clips: {result['clip_count']}")

    if extract:
        extract_dir = target_root / str(result["run_id"])
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(result["archive_bytes"])) as archive:
            archive.extractall(extract_dir)
        print(f"extracted artifacts: {extract_dir}")


def _load_json_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(config_path: Path, configured_output_dir: str | None) -> Path:
    output_dir = Path(configured_output_dir or "runs/encodec_demo")
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    return output_dir


def _zip_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return buffer.getvalue()
