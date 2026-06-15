from __future__ import annotations

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
