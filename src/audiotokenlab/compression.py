from __future__ import annotations

import json
from pathlib import Path

from audiotokenlab.models import TokenBundle


def compress_tokens(bundle: TokenBundle, strategy: dict) -> TokenBundle:
    name = strategy.get("name", "baseline")
    strategy_label = str(strategy.get("label", name))
    if name == "baseline":
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy_label)
    if name == "uniform":
        factor = int(strategy.get("factor", 2))
        return _uniform(bundle, factor=factor, strategy=strategy_label)
    if name == "silence_aware":
        factor = int(strategy.get("factor", 2))
        threshold = int(strategy.get("threshold", 8))
        threshold_low = strategy.get("threshold_low")
        threshold_high = strategy.get("threshold_high")
        return _silence_aware(
            bundle,
            factor=factor,
            threshold=threshold,
            threshold_low=int(threshold_low) if threshold_low is not None else None,
            threshold_high=int(threshold_high) if threshold_high is not None else None,
            strategy=strategy_label,
        )
    if name == "patch":
        patch_size = int(strategy.get("patch_size", 4))
        return _patch(bundle, patch_size=patch_size, strategy=strategy_label)
    if name == "acoustic_salience":
        factor = int(strategy.get("factor", 2))
        return _acoustic_salience(bundle, factor=factor, strategy=strategy_label)
    if name == "energy_salience":
        factor = int(strategy.get("factor", 2))
        return _energy_salience(
            bundle,
            factor=factor,
            energy_weight=float(strategy.get("energy_weight", 2.0)),
            transition_weight=float(strategy.get("transition_weight", 1.0)),
            onset_weight=float(strategy.get("onset_weight", 2.0)),
            silence_threshold=float(strategy.get("silence_threshold", 0.05)),
            strategy=strategy_label,
        )
    if name == "vad_salience":
        factor = int(strategy.get("factor", 2))
        return _vad_salience(
            bundle,
            factor=factor,
            noise_floor_ratio=float(strategy.get("noise_floor_ratio", 1.8)),
            absolute_threshold=float(strategy.get("absolute_threshold", 0.04)),
            min_speech_frames=int(strategy.get("min_speech_frames", 2)),
            hangover_frames=int(strategy.get("hangover_frames", 1)),
            transition_weight=float(strategy.get("transition_weight", 1.0)),
            onset_weight=float(strategy.get("onset_weight", 2.0)),
            strategy=strategy_label,
        )
    if name == "learned_selector":
        factor = int(strategy.get("factor", 2))
        return _learned_selector(
            bundle,
            factor=factor,
            weights=_selector_weights(strategy),
            strategy=strategy_label,
        )
    raise ValueError(f"Unsupported compression strategy: {name}")


def _replace(
    bundle: TokenBundle,
    tokens: tuple[int, ...],
    strategy: str,
    extra_metadata: dict | None = None,
) -> TokenBundle:
    metadata = dict(bundle.metadata)
    metadata["compression_strategy"] = strategy
    if extra_metadata:
        metadata.update(extra_metadata)
    if _uses_frame_groups(bundle):
        metadata["frame_count"] = len(tokens) // bundle.codebook_count
    return TokenBundle(
        clip_id=bundle.clip_id,
        tokenizer=bundle.tokenizer,
        tokens=tokens,
        frame_rate=bundle.frame_rate,
        codebook_count=bundle.codebook_count,
        sample_rate=bundle.sample_rate,
        duration_seconds=bundle.duration_seconds,
        metadata=metadata,
    )


def _uniform(bundle: TokenBundle, factor: int, strategy: str) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)
    if _uses_frame_groups(bundle):
        return _replace(
            bundle,
            tokens=_flatten_frames(_frame_groups(bundle)[::factor]),
            strategy=strategy,
        )
    return _replace(bundle, tokens=tuple(bundle.tokens[::factor]), strategy=strategy)


def _silence_aware(
    bundle: TokenBundle,
    factor: int,
    threshold: int,
    threshold_low: int | None,
    threshold_high: int | None,
    strategy: str,
) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)

    if _uses_frame_groups(bundle):
        kept_frames: list[tuple[int, ...]] = []
        quiet_seen = 0
        for frame in _frame_groups(bundle):
            if all(
                _is_quiet_token(token, threshold, threshold_low, threshold_high)
                for token in frame
            ):
                if quiet_seen % factor == 0:
                    kept_frames.append(frame)
                quiet_seen += 1
            else:
                kept_frames.append(frame)
                quiet_seen = 0
        return _replace(bundle, tokens=_flatten_frames(kept_frames), strategy=strategy)

    kept: list[int] = []
    quiet_seen = 0
    for token in bundle.tokens:
        if _is_quiet_token(token, threshold, threshold_low, threshold_high):
            if quiet_seen % factor == 0:
                kept.append(token)
            quiet_seen += 1
        else:
            kept.append(token)
            quiet_seen = 0
    return _replace(bundle, tokens=tuple(kept), strategy=strategy)


def _is_quiet_token(
    token: int,
    threshold: int,
    threshold_low: int | None,
    threshold_high: int | None,
) -> bool:
    if threshold_low is not None and threshold_high is not None:
        return threshold_low <= token <= threshold_high
    return token <= threshold


def _patch(bundle: TokenBundle, patch_size: int, strategy: str) -> TokenBundle:
    if patch_size <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)

    if _uses_frame_groups(bundle):
        patched_frames: list[tuple[int, ...]] = []
        frames = _frame_groups(bundle)
        for start in range(0, len(frames), patch_size):
            patch = frames[start : start + patch_size]
            if not patch:
                continue
            merged: list[int] = []
            for codebook_index in range(bundle.codebook_count):
                merged.append(
                    round(
                        sum(frame[codebook_index] for frame in patch) / len(patch)
                    )
                )
            patched_frames.append(tuple(merged))
        return _replace(bundle, tokens=_flatten_frames(patched_frames), strategy=strategy)

    patched: list[int] = []
    for start in range(0, len(bundle.tokens), patch_size):
        patch = bundle.tokens[start : start + patch_size]
        if patch:
            patched.append(round(sum(patch) / len(patch)))
    return _replace(bundle, tokens=tuple(patched), strategy=strategy)


def _acoustic_salience(bundle: TokenBundle, factor: int, strategy: str) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)
    if not _uses_frame_groups(bundle):
        return _replace(bundle, tokens=tuple(bundle.tokens[::factor]), strategy=strategy)

    frames = _frame_groups(bundle)
    if not frames:
        return _replace(bundle, tokens=(), strategy=strategy)

    kept_frames: list[tuple[int, ...]] = []
    repeat_counts: list[int] = []
    for start in range(0, len(frames), factor):
        window = frames[start : start + factor]
        best_offset = _highest_salience_offset(frames, start, len(window))
        kept_frames.append(window[best_offset])
        repeat_counts.append(len(window))

    return _replace(
        bundle,
        tokens=_flatten_frames(kept_frames),
        strategy=strategy,
        extra_metadata={
            "decode_repeat_counts": repeat_counts,
            "decode_frame_count": len(frames),
            "salience_factor": factor,
        },
    )


def _energy_salience(
    bundle: TokenBundle,
    factor: int,
    energy_weight: float,
    transition_weight: float,
    onset_weight: float,
    silence_threshold: float,
    strategy: str,
) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)
    if not _uses_frame_groups(bundle):
        return _replace(bundle, tokens=tuple(bundle.tokens[::factor]), strategy=strategy)

    frames = _frame_groups(bundle)
    if not frames:
        return _replace(bundle, tokens=(), strategy=strategy)

    energies = _metadata_float_list(bundle.metadata.get("frame_energies"))
    if len(energies) != len(frames):
        return _acoustic_salience(bundle, factor=factor, strategy=strategy)

    kept_frames: list[tuple[int, ...]] = []
    repeat_counts: list[int] = []
    normalized = _normalize_values(energies)
    threshold = max(normalized) * silence_threshold if normalized else 0.0
    for start in range(0, len(frames), factor):
        window = frames[start : start + factor]
        best_offset = _highest_energy_salience_offset(
            frames,
            normalized,
            start,
            len(window),
            energy_weight=energy_weight,
            transition_weight=transition_weight,
            onset_weight=onset_weight,
            silence_threshold=threshold,
        )
        kept_frames.append(window[best_offset])
        repeat_counts.append(len(window))

    return _replace(
        bundle,
        tokens=_flatten_frames(kept_frames),
        strategy=strategy,
        extra_metadata={
            "decode_repeat_counts": repeat_counts,
            "decode_frame_count": len(frames),
            "salience_factor": factor,
            "energy_weight": energy_weight,
            "transition_weight": transition_weight,
            "onset_weight": onset_weight,
            "silence_threshold": silence_threshold,
        },
    )


def _vad_salience(
    bundle: TokenBundle,
    factor: int,
    noise_floor_ratio: float,
    absolute_threshold: float,
    min_speech_frames: int,
    hangover_frames: int,
    transition_weight: float,
    onset_weight: float,
    strategy: str,
) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)
    if not _uses_frame_groups(bundle):
        return _replace(bundle, tokens=tuple(bundle.tokens[::factor]), strategy=strategy)

    frames = _frame_groups(bundle)
    if not frames:
        return _replace(bundle, tokens=(), strategy=strategy)

    energies = _metadata_float_list(bundle.metadata.get("frame_energies"))
    if len(energies) != len(frames):
        return _acoustic_salience(bundle, factor=factor, strategy=strategy)

    normalized = _normalize_values(energies)
    speech_mask = _speech_activity_mask(
        normalized,
        noise_floor_ratio=noise_floor_ratio,
        absolute_threshold=absolute_threshold,
        min_speech_frames=min_speech_frames,
        hangover_frames=hangover_frames,
    )
    kept_frames: list[tuple[int, ...]] = []
    repeat_counts: list[int] = []
    for start in range(0, len(frames), factor):
        window = frames[start : start + factor]
        best_offset = _highest_vad_salience_offset(
            frames,
            normalized,
            speech_mask,
            start,
            len(window),
            transition_weight=transition_weight,
            onset_weight=onset_weight,
        )
        kept_frames.append(window[best_offset])
        repeat_counts.append(len(window))

    return _replace(
        bundle,
        tokens=_flatten_frames(kept_frames),
        strategy=strategy,
        extra_metadata={
            "decode_repeat_counts": repeat_counts,
            "decode_frame_count": len(frames),
            "salience_factor": factor,
            "speech_activity_ratio": sum(1 for item in speech_mask if item) / len(speech_mask),
            "noise_floor_ratio": noise_floor_ratio,
            "absolute_threshold": absolute_threshold,
            "min_speech_frames": min_speech_frames,
            "hangover_frames": hangover_frames,
        },
    )


def _learned_selector(
    bundle: TokenBundle,
    factor: int,
    weights: dict,
    strategy: str,
) -> TokenBundle:
    if factor <= 1:
        return _replace(bundle, tokens=bundle.tokens, strategy=strategy)
    if not _uses_frame_groups(bundle):
        return _replace(bundle, tokens=tuple(bundle.tokens[::factor]), strategy=strategy)

    frames = _frame_groups(bundle)
    if not frames:
        return _replace(bundle, tokens=(), strategy=strategy)

    energies = _metadata_float_list(bundle.metadata.get("frame_energies"))
    if len(energies) != len(frames):
        energies = [0.0 for _ in frames]
    normalized = _normalize_values(energies)
    speech_mask = _speech_activity_mask(
        normalized,
        noise_floor_ratio=float(weights.get("noise_floor_ratio", 1.8)),
        absolute_threshold=float(weights.get("absolute_threshold", 0.04)),
        min_speech_frames=int(weights.get("min_speech_frames", 1)),
        hangover_frames=int(weights.get("hangover_frames", 1)),
    )
    kept_frames: list[tuple[int, ...]] = []
    repeat_counts: list[int] = []
    for start in range(0, len(frames), factor):
        window = frames[start : start + factor]
        best_offset = _highest_learned_selector_offset(
            frames,
            normalized,
            speech_mask,
            start,
            len(window),
            weights=weights,
        )
        kept_frames.append(window[best_offset])
        repeat_counts.append(len(window))

    return _replace(
        bundle,
        tokens=_flatten_frames(kept_frames),
        strategy=strategy,
        extra_metadata={
            "decode_repeat_counts": repeat_counts,
            "decode_frame_count": len(frames),
            "salience_factor": factor,
            "selector_type": "linear_frame_selector",
            "selector_weights": dict(weights),
        },
    )


def _highest_salience_offset(
    frames: list[tuple[int, ...]],
    start: int,
    window_size: int,
) -> int:
    best_offset = 0
    best_score = -1.0
    for offset in range(window_size):
        frame_index = start + offset
        previous_frame = frames[frame_index - 1] if frame_index > 0 else frames[frame_index]
        score = _frame_transition_score(previous_frame, frames[frame_index])
        if score > best_score:
            best_offset = offset
            best_score = score
    return best_offset


def _highest_vad_salience_offset(
    frames: list[tuple[int, ...]],
    energies: list[float],
    speech_mask: list[bool],
    start: int,
    window_size: int,
    transition_weight: float,
    onset_weight: float,
) -> int:
    best_offset = 0
    best_score = -1.0
    for offset in range(window_size):
        frame_index = start + offset
        previous_frame = frames[frame_index - 1] if frame_index > 0 else frames[frame_index]
        previous_speech = speech_mask[frame_index - 1] if frame_index > 0 else speech_mask[frame_index]
        speech_score = 1.0 if speech_mask[frame_index] else 0.0
        onset_score = 1.0 if not previous_speech and speech_mask[frame_index] else 0.0
        transition_score = _frame_transition_score(previous_frame, frames[frame_index])
        score = (
            speech_score
            + energies[frame_index]
            + transition_weight * transition_score
            + onset_weight * onset_score
        )
        if score > best_score:
            best_offset = offset
            best_score = score
    return best_offset


def _highest_learned_selector_offset(
    frames: list[tuple[int, ...]],
    energies: list[float],
    speech_mask: list[bool],
    start: int,
    window_size: int,
    weights: dict,
) -> int:
    learned_weights = {
        "bias": -0.05,
        "energy": 2.2,
        "onset": 1.4,
        "transition": 1.0,
        "speech_activity": 1.8,
        "center": 0.15,
    }
    learned_weights.update({key: float(value) for key, value in weights.items() if _is_number(value)})
    best_offset = 0
    best_score = -1e9
    center = (window_size - 1) / 2.0
    for offset in range(window_size):
        frame_index = start + offset
        previous_frame = frames[frame_index - 1] if frame_index > 0 else frames[frame_index]
        previous_energy = energies[frame_index - 1] if frame_index > 0 else energies[frame_index]
        transition_score = _frame_transition_score(previous_frame, frames[frame_index])
        onset_score = max(0.0, energies[frame_index] - previous_energy)
        speech_score = 1.0 if speech_mask[frame_index] else 0.0
        center_score = 1.0 - abs(offset - center) / max(1.0, center + 1.0)
        score = (
            learned_weights["bias"]
            + learned_weights["energy"] * energies[frame_index]
            + learned_weights["onset"] * onset_score
            + learned_weights["transition"] * transition_score
            + learned_weights["speech_activity"] * speech_score
            + learned_weights["center"] * center_score
        )
        if score > best_score:
            best_offset = offset
            best_score = score
    return best_offset


def _highest_energy_salience_offset(
    frames: list[tuple[int, ...]],
    energies: list[float],
    start: int,
    window_size: int,
    energy_weight: float,
    transition_weight: float,
    onset_weight: float,
    silence_threshold: float,
) -> int:
    best_offset = 0
    best_score = -1.0
    for offset in range(window_size):
        frame_index = start + offset
        previous_frame = frames[frame_index - 1] if frame_index > 0 else frames[frame_index]
        previous_energy = energies[frame_index - 1] if frame_index > 0 else energies[frame_index]
        transition_score = _frame_transition_score(previous_frame, frames[frame_index])
        onset_score = (
            1.0
            if previous_energy <= silence_threshold < energies[frame_index]
            else 0.0
        )
        score = (
            energy_weight * energies[frame_index]
            + transition_weight * transition_score
            + onset_weight * onset_score
        )
        if score > best_score:
            best_offset = offset
            best_score = score
    return best_offset


def _frame_transition_score(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    changed = 0
    magnitude = 0
    count = min(len(left), len(right))
    for index in range(count):
        delta = abs(left[index] - right[index])
        if delta:
            changed += 1
            magnitude += delta
    return changed + (magnitude / max(1, count * 1024))


def _metadata_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= 0.0:
        return [0.0 for _ in values]
    return [value / max_value for value in values]


def _speech_activity_mask(
    energies: list[float],
    noise_floor_ratio: float,
    absolute_threshold: float,
    min_speech_frames: int,
    hangover_frames: int,
) -> list[bool]:
    if not energies:
        return []
    sorted_energy = sorted(energies)
    median = sorted_energy[len(sorted_energy) // 2]
    threshold = max(absolute_threshold, median * noise_floor_ratio)
    raw_mask = [energy >= threshold for energy in energies]
    mask = _remove_short_runs(raw_mask, min_speech_frames)
    if hangover_frames > 0:
        expanded = list(mask)
        for index, active in enumerate(mask):
            if not active:
                continue
            start = max(0, index - hangover_frames)
            end = min(len(expanded), index + hangover_frames + 1)
            for expanded_index in range(start, end):
                expanded[expanded_index] = True
        mask = expanded
    return mask


def _remove_short_runs(mask: list[bool], min_length: int) -> list[bool]:
    if min_length <= 1:
        return list(mask)
    cleaned = list(mask)
    start = 0
    while start < len(mask):
        value = mask[start]
        end = start + 1
        while end < len(mask) and mask[end] == value:
            end += 1
        if value and end - start < min_length:
            for index in range(start, end):
                cleaned[index] = False
        start = end
    return cleaned


def _is_number(value: object) -> bool:
    return isinstance(value, int | float)


def _selector_weights(strategy: dict) -> dict:
    weights = dict(strategy.get("weights", {}))
    weights_path = strategy.get("weights_path")
    if weights_path:
        loaded = json.loads(Path(str(weights_path)).read_text(encoding="utf-8"))
        if "trained_strategy" in loaded:
            loaded = loaded["trained_strategy"]
        weights.update(dict(loaded.get("weights", loaded)))
    return weights


def _uses_frame_groups(bundle: TokenBundle) -> bool:
    return (
        bundle.metadata.get("token_layout") == "frame_major"
        and bundle.codebook_count > 1
        and len(bundle.tokens) % bundle.codebook_count == 0
    )


def _frame_groups(bundle: TokenBundle) -> list[tuple[int, ...]]:
    return [
        bundle.tokens[start : start + bundle.codebook_count]
        for start in range(0, len(bundle.tokens), bundle.codebook_count)
    ]


def _flatten_frames(frames: list[tuple[int, ...]]) -> tuple[int, ...]:
    flattened: list[int] = []
    for frame in frames:
        flattened.extend(frame)
    return tuple(flattened)
