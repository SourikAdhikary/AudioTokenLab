from __future__ import annotations

from audiotokenlab.models import TokenBundle


def compress_tokens(bundle: TokenBundle, strategy: dict) -> TokenBundle:
    name = strategy.get("name", "baseline")
    if name == "baseline":
        return _replace(bundle, tokens=bundle.tokens, strategy=name)
    if name == "uniform":
        factor = int(strategy.get("factor", 2))
        return _uniform(bundle, factor=factor, strategy=name)
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
            strategy=name,
        )
    if name == "patch":
        patch_size = int(strategy.get("patch_size", 4))
        return _patch(bundle, patch_size=patch_size, strategy=name)
    raise ValueError(f"Unsupported compression strategy: {name}")


def _replace(bundle: TokenBundle, tokens: tuple[int, ...], strategy: str) -> TokenBundle:
    metadata = dict(bundle.metadata)
    metadata["compression_strategy"] = strategy
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

    patched: list[int] = []
    for start in range(0, len(bundle.tokens), patch_size):
        patch = bundle.tokens[start : start + patch_size]
        if patch:
            patched.append(round(sum(patch) / len(patch)))
    return _replace(bundle, tokens=tuple(patched), strategy=strategy)
