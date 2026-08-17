"""Streams image-text pairs straight out of parquet shards (CC3M, or any img2dataset export):
each row bundles the JPEG bytes and a caption, so no parquet->tar conversion is needed.

    ParquetDataModule(train_shards="data/cc3m/data/*.parquet", num_train_samples=3016640)"""

from __future__ import annotations

import functools
import glob
import io
import math
import os
from typing import Optional

import lightning as L
import pyarrow.parquet as pq
import torch.distributed as dist
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.data.processing import encode_image_text
from src.data.webdataset_module import _collate

Image.MAX_IMAGE_PIXELS = None

# non-caption columns read from every shard (only those present are requested).
_BASE_COLUMNS = ["status", "clip_similarity_vitl14"]
_CAPTION_COL = "caption"
# image byte column, in preference order. img2dataset shards store a top-level "jpg" binary column;
# HF-style shards (e.g. WeiChow/cc3m) store an "image" struct {bytes, path}.
_IMAGE_COLS = ["jpg", "image"]


def _img_bytes(v):
    """extract JPEG bytes from a cell that is either raw bytes (jpg col) or an
    HF image struct {"bytes":..., "path":...} (image col)."""
    if isinstance(v, dict):
        return v.get("bytes")
    return v


def _as_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    v = v.strip()
    return v or None


def _expand_parquet_shards(pattern: str | list[str] | None) -> list[str]:
    """expand a glob (or list of globs) to a sorted list of parquet shard paths."""
    if pattern is None:
        return []
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths: list[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return sorted(set(paths))


_DROP_SET_CACHE: dict[str, frozenset] = {}


def _load_drop_set(path: str) -> frozenset:
    """Exact-caption dedup drop list: the row `id`s to skip during training. One int64 column
    ``drop_id``, holding every later row whose caption is byte-identical to an earlier kept row's.
    Loaded once per worker and cached."""
    if path not in _DROP_SET_CACHE:
        t = pq.read_table(path, columns=["drop_id"])
        _DROP_SET_CACHE[path] = frozenset(int(x) for x in t.column("drop_id").to_pylist())
    return _DROP_SET_CACHE[path]


def _iter_parquet_shard(src, *, row_batch_size: int = 256, min_clip_sim: float | None = None,
                        caption_col: str = _CAPTION_COL,
                        caption_dedup_drop_table: str | None = None):
    """pipeline stage: shard specs -> {"jpg": bytes, "txt": str}.

    ``min_clip_sim`` skips rows whose clip_similarity_vitl14 is below the threshold (rows missing
    the column pass through). ``caption_col`` selects the text column. The image column is
    auto-detected: a top-level "jpg" binary or an "image" struct {bytes, path}.

    ``caption_dedup_drop_table``: parquet drop list of row `id`s to skip. A shard with no `id`
    column raises rather than silently training the full un-deduped corpus."""
    drop = _load_drop_set(caption_dedup_drop_table) if caption_dedup_drop_table else None
    for shard in src:
        path = shard["url"] if isinstance(shard, dict) else shard
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        # top-level arrow field names: schema.names reports physical nested paths, hiding struct
        # columns behind 'image.bytes'/'image.path' and list columns behind an 'element' leaf.
        available = set(n.split(".")[0] for n in pf.schema_arrow.names)
        img_col = next((c for c in _IMAGE_COLS if c in available), None)
        if img_col is None:
            continue
        read_cols = [c for c in _BASE_COLUMNS if c in available] + [img_col]
        has_cap = caption_col in available
        if has_cap:
            read_cols.append(caption_col)
        has_drop = drop is not None
        if has_drop:
            if "id" not in available:
                raise ValueError(
                    f"caption_dedup_drop_table set but shard {path!r} has no 'id' column — cannot "
                    f"apply the exact-caption drop list (available: {sorted(available)})")
            if "id" not in read_cols:
                read_cols.append("id")
        for batch in pf.iter_batches(batch_size=row_batch_size, columns=read_cols):
            cols = batch.to_pydict()
            imgs = cols[img_col]
            n = len(imgs)
            stats = cols.get("status", [None] * n)
            sims = cols.get("clip_similarity_vitl14", [None] * n)
            for i in range(n):
                jpg = _img_bytes(imgs[i])
                if jpg is None:
                    continue
                if stats[i] is not None and stats[i] not in ("success", b"success"):
                    continue
                if min_clip_sim is not None and sims[i] is not None and sims[i] < min_clip_sim:
                    continue
                if has_drop:
                    try:
                        did = int(cols["id"][i])   # the CC3M id is a string; fail loud if uncastable
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"caption_dedup_drop_table active but row id in {path!r} is "
                            f"missing/non-castable: {cols['id'][i]!r}")
                    if did in drop:
                        continue
                txt = _as_str(cols[caption_col][i]) if has_cap else None
                if txt is None:
                    continue
                yield {"jpg": jpg, "txt": txt}


def _make_decoder(processor: AutoProcessor, max_length: int):
    """the per-sample decode + tokenize map fn."""
    def _decode(sample):
        with Image.open(io.BytesIO(sample["jpg"])) as im:
            im.load()
            if im.width < 10 or im.height < 10:
                raise ValueError("Image too small")
            if im.mode != "RGB":
                im = im.convert("RGB")
            return encode_image_text(processor, im, sample["txt"], max_length)

    return _decode


def _make_pipeline(
    shards: list[str],
    processor: AutoProcessor,
    max_length: int,
    shuffle: int,
    is_train: bool,
    batch_size: int,
    min_clip_sim: float | None = None,
    caption_col: str = _CAPTION_COL,
    caption_dedup_drop_table: str | None = None,   # exact-caption dedup drop list path; TRAIN only
) -> wds.DataPipeline:
    """build the shard->batch pipeline. train uses i.i.d. shard resampling (reseeded per
    worker/rank/epoch) + a sample buffer; val/test use an ordered split across node/worker."""
    decode = wds.map(_make_decoder(processor, max_length), handler=wds.warn_and_continue)
    iter_shard = functools.partial(
        _iter_parquet_shard, min_clip_sim=min_clip_sim, caption_col=caption_col,
        caption_dedup_drop_table=(caption_dedup_drop_table if is_train else None),
    )
    if is_train:
        stages = [wds.ResampledShards(shards), iter_shard]
        if shuffle > 0:
            stages.append(wds.shuffle(shuffle))
        stages.append(decode)
        return wds.DataPipeline(
            wds.DataPipeline(*stages),
            wds.batched(batch_size, collation_fn=_collate, partial=False),
        )
    return wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_node,
        wds.split_by_worker,
        iter_shard,
        decode,
        wds.batched(batch_size, collation_fn=_collate, partial=True),
    )


class ParquetDataModule(L.LightningDataModule):
    """LightningDataModule for parquet image-text shards."""

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
        compute_retrieval_metrics: bool = False,
        min_clip_sim: float | None = None,
        caption_col: str = _CAPTION_COL,
        caption_dedup_drop_table: Optional[str] = None,    # exact-caption dedup drop list; TRAIN only
    ):
        super().__init__()
        self.save_hyperparameters()
        self.compute_retrieval_metrics = compute_retrieval_metrics
        self.min_clip_sim = min_clip_sim
        self.caption_col = caption_col
        self.caption_dedup_drop_table = caption_dedup_drop_table
        if caption_dedup_drop_table and caption_col != _CAPTION_COL:
            raise ValueError(
                f"caption_dedup_drop_table is keyed on the plain {_CAPTION_COL!r} column and "
                f"cannot be combined with caption_col={caption_col!r} — build a dedicated drop "
                f"list for that text source.")

        self.model_name = model_name
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples or num_val_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_factor = prefetch_factor

        self._train_shards = _expand_parquet_shards(train_shards)
        self._val_shards = _expand_parquet_shards(val_shards)
        self._test_shards = _expand_parquet_shards(test_shards) or self._val_shards

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def _world_size(self) -> int:
        if hasattr(self, "target_world_size"):
            return self.target_world_size
        if dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", 1))

    def _validate_caption_col(self, shards: list[str]) -> None:
        """fail loudly on a mistyped/absent caption_col (else the reader silently yields an
        EMPTY dataset — every row is dropped for a missing caption). Peek the first shard."""
        for s in shards:
            try:
                avail = set(n.split(".")[0] for n in pq.ParquetFile(s).schema_arrow.names)
            except Exception:
                continue
            if self.caption_col not in avail:
                raise ValueError(
                    f"caption_col={self.caption_col!r} not found in shard schema {sorted(avail)} "
                    f"({s}). Set caption_col to an existing text column.")
            return

    def _make_loader(self, shards: list[str], num_samples: int, is_train: bool) -> DataLoader:
        self._validate_caption_col(shards)
        world_size = self._world_size()
        pipeline = _make_pipeline(
            shards=shards,
            processor=self.processor,
            max_length=self.max_length,
            shuffle=self.shuffle_buffer if is_train else 0,
            is_train=is_train,
            batch_size=self.batch_size,
            min_clip_sim=self.min_clip_sim,
            caption_col=self.caption_col,
            caption_dedup_drop_table=self.caption_dedup_drop_table,
        )

        local_samples = max(1, num_samples // world_size)
        if is_train:
            num_batches = max(1, local_samples // self.batch_size)
        else:
            num_batches = max(1, math.ceil(local_samples / self.batch_size))

        # with_epoch runs per worker, so divide by num_workers to keep the per-rank
        # epoch length == num_batches.
        n_workers = max(1, self.num_workers)
        per_worker_batches = max(1, num_batches // n_workers)
        effective_batches = per_worker_batches * n_workers
        pipeline = pipeline.with_epoch(per_worker_batches)

        loader = wds.WebLoader(
            pipeline,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
        return loader.with_length(effective_batches)

    def train_dataloader(self):
        if not self._train_shards:
            raise FileNotFoundError(
                f"no parquet shards matched train_shards={self.hparams.train_shards!r}")
        return self._make_loader(self._train_shards, self.num_train_samples, is_train=True)

    def val_dataloader(self):
        if not self._val_shards:
            return []
        return self._make_loader(self._val_shards, self.num_val_samples, is_train=False)

    def test_dataloader(self):
        if not self._test_shards:
            return []
        return self._make_loader(self._test_shards, self.num_test_samples, is_train=False)
