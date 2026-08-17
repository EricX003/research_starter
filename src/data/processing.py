"""Shared image-text encoding helpers."""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor


def encode_image_text(
    processor: AutoProcessor,
    image,
    caption: str,
    max_length: int,
) -> dict[str, Any]:
    """Encode one image-caption pair into tensors for training.

    Calls ``image_processor`` and ``tokenizer`` separately so we can use
    ``use_fast=True`` without hitting CLIPProcessor's broken joint ``__call__``
    (fast processors lack ``_valid_processor_keys``).
    """
    image_out = processor.image_processor(images=image, return_tensors="pt")
    text_out = processor.tokenizer(
        text=caption,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,   # SigLIP tokenizers omit it by default; force it (no-op for CLIP)
    )
    return {
        "pixel_values": image_out["pixel_values"].squeeze(0),
        "input_ids": text_out["input_ids"].squeeze(0),
        "attention_mask": text_out["attention_mask"].squeeze(0),
    }
