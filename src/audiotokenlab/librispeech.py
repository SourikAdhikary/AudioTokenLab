from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_LIBRISPEECH_DEV_CLEAN_URL = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
)


def prepare_librispeech_slice(
    output_dir: Path,
    archive_url: str = DEFAULT_LIBRISPEECH_DEV_CLEAN_URL,
    max_clips: int = 4,
    sample_rate: int = 24000,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "dev-clean.tar.gz"
    flac_dir = output_dir / "flac"
    wav_dir = output_dir / "wav"
    flac_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        urllib.request.urlretrieve(archive_url, archive_path)

    transcripts: dict[str, str] = {}
    selected_flacs: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.isfile() and member.name.endswith(".trans.txt"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read().decode("utf-8")
                transcripts.update(parse_librispeech_transcripts(text))

        for member in members:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            clip_id = Path(member.name).stem
            if clip_id not in transcripts:
                continue
            selected_flacs.append(member)
            if len(selected_flacs) >= max_clips:
                break

        manifest_clips = []
        for member in selected_flacs:
            clip_id = Path(member.name).stem
            flac_path = flac_dir / f"{clip_id}.flac"
            wav_path = wav_dir / f"{clip_id}.wav"
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with flac_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(flac_path),
                    "-ar",
                    str(sample_rate),
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
                    "transcript": transcripts[clip_id],
                    "source": "LibriSpeech dev-clean",
                    "license": "CC BY 4.0",
                    "archive_url": archive_url,
                }
            )

    if not manifest_clips:
        raise ValueError("No LibriSpeech clips were extracted")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"root": str(wav_dir), "clips": manifest_clips}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def parse_librispeech_transcripts(text: str) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        clip_id, _, transcript = line.partition(" ")
        if clip_id and transcript:
            transcripts[clip_id] = transcript
    return transcripts

