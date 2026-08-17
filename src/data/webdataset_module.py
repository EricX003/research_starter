"""Data module for tar-sharded image-text corpora, where each shard holds ``{key}.jpg`` and
``{key}.txt`` per sample."""

from __future__ import annotations

import glob
import math
import random
import re
import os
from typing import Optional

import lightning as L
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.data.processing import encode_image_text


def _expand_shards(pattern: str) -> list[str]:
    """Expand a WebDataset shard pattern to a sorted list of paths.

    Handles WebDataset's numeric range syntax, e.g.
        /data/shards-{0000..0015}.tar  ->  ['/data/shards-0000.tar', ...]
    Falls back to glob for wildcard patterns.
    """
    if pattern is None:
        return []

    match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if match:
        width = len(match.group(1))
        start, end = int(match.group(1)), int(match.group(2))
        prefix, suffix = pattern[:match.start()], pattern[match.end():]
        return [f"{prefix}{str(i).zfill(width)}{suffix}" for i in range(start, end + 1)]
    return sorted(glob.glob(pattern))


def _make_pipeline(
    shards: list[str] | str,
    processor: AutoProcessor,
    max_length: int,
    batch_size: int,
    shuffle: int,
    num_samples: int,
    is_train: bool,
) -> wds.WebDataset:
    pipeline = wds.WebDataset(
        shards,
        shardshuffle=1000 if is_train else False,
        handler=wds.ignore_and_continue,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    )

    if is_train:
        pipeline = pipeline.shuffle(shuffle) if shuffle > 0 else pipeline

    def _preprocess(sample):
        image = sample["jpg"]
        caption = sample["txt"].decode("utf-8") if isinstance(sample["txt"], bytes) else sample["txt"]

        if image.width < 10 or image.height < 10:
            raise ValueError("Image too small")

        if image.mode != "RGB":
            image = image.convert("RGB")

        return encode_image_text(processor, image, caption, max_length)

    pipeline = (
        pipeline
        .decode("pil", handler=wds.warn_and_continue)
        .map(_preprocess, handler=wds.warn_and_continue)
    )

    return pipeline


class WebDatasetModule(L.LightningDataModule):
    """LightningDataModule for tar-sharded WebDataset image-text pairs."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        train_shards: Optional[str] = None,
        val_shards: Optional[str] = None,
        test_shards: Optional[str] = None,
        num_train_samples: int = 0,
        num_val_samples: int = 0,
        num_test_samples: int = 0,
        max_length: int = 77,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_buffer: int = 5000,
        prefetch_factor: int = 2,
        val_num_shards: Optional[int] = None,
        compute_retrieval_metrics: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.compute_retrieval_metrics = compute_retrieval_metrics

        self.model_name = model_name
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards or val_shards
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples or num_val_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_factor = prefetch_factor

        self._all_val_shards = _expand_shards(val_shards)
        self.val_num_shards = val_num_shards  # None = use all

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def _make_loader(
        self,
        shards: str,
        num_samples: int,
        is_train: bool,
    ) -> DataLoader:
        if hasattr(self, "target_world_size"):
            world_size = self.target_world_size
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        pipeline = _make_pipeline(
            shards=shards,
            processor=self.processor,
            max_length=self.max_length,
            batch_size=self.batch_size,
            shuffle=self.shuffle_buffer if is_train else 0,
            num_samples=num_samples,
            is_train=is_train,
        )

        batched = pipeline.batched(self.batch_size, collation_fn=_collate, partial=not is_train)
        local_samples = num_samples // world_size
        num_batches = math.ceil(local_samples / self.batch_size) if not is_train else local_samples // self.batch_size

        batched = batched.with_epoch(num_batches)

        loader = wds.WebLoader(
            batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

        loader = loader.with_length(num_batches)
        return loader

    def train_dataloader(self):
        return self._make_loader(self.train_shards, self.num_train_samples, is_train=True)

    def val_dataloader(self):
        if not self.val_shards or not self._all_val_shards:
            return []

        total = len(self._all_val_shards)
        if self.val_num_shards is not None and self.val_num_shards < total:
            shards = random.sample(self._all_val_shards, self.val_num_shards)
            num_samples = round(self.num_val_samples * self.val_num_shards / total)
        else:
            shards = self._all_val_shards
            num_samples = self.num_val_samples
        return self._make_loader(shards, num_samples, is_train=False)

    def test_dataloader(self):
        if not self.test_shards:
            return []
        return self._make_loader(self.test_shards, self.num_test_samples, is_train=False)


_COLLATE_KEYS = ("pixel_values", "input_ids", "attention_mask")


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([sample[k] for sample in batch]) for k in _COLLATE_KEYS}
