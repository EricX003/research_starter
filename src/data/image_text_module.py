"""Image-text pair data module backed by HuggingFace datasets; tokenization and image
preprocessing come from the backbone's ``AutoProcessor``."""

from __future__ import annotations

from typing import Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.data.processing import encode_image_text


def _extract_caption(value) -> str:
    """Normalise heterogeneous HF caption fields to a single string.

    Handles:
      str                                — returned as-is
      list[str] / list[dict]             — first element
      dict with ``"raw"`` key            — value["raw"]
      list of dicts with ``"raw"`` keys  — first element's ``"raw"``
    """
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, dict) and "raw" in value:
        value = value["raw"]
    if not isinstance(value, str):
        raise TypeError(f"Could not extract caption string; got {type(value).__name__}: {value!r}")
    return value


class ImageTextDataset:
    """Wraps a HuggingFace dataset split, applying processor on-the-fly."""

    def __init__(self, hf_dataset, processor, image_column, caption_column, max_length):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        caption = _extract_caption(item[self.caption_column])

        if image.mode != "RGB":
            image = image.convert("RGB")

        batch_dict = encode_image_text(self.processor, image, caption, self.max_length)
        batch_dict["sample_idx"] = idx

        return batch_dict


class ImageTextDataModule(L.LightningDataModule):
    """LightningDataModule that loads image-text pairs from HuggingFace Hub or local disk.

    Config parameters map directly to ``datasets.load_dataset`` and
    ``transformers.AutoProcessor``.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dataset_name: str = "nlphuji/flickr30k",
        dataset_config: Optional[str] = None,
        image_column: str = "image",
        caption_column: str = "caption",
        train_split: str = "test",
        val_split: str = "test",
        test_split: str = "test",
        filter_column: Optional[str] = None,
        train_filter_value: Optional[str] = None,
        val_filter_value: Optional[str] = None,
        test_filter_value: Optional[str] = None,
        max_length: int = 77,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        compute_retrieval_metrics: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.image_column = image_column
        self.caption_column = caption_column
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.filter_column = filter_column
        self.train_filter_value = train_filter_value
        self.val_filter_value = val_filter_value
        self.test_filter_value = test_filter_value
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.compute_retrieval_metrics = compute_retrieval_metrics

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def setup(self, stage: Optional[str] = None):
        load_kw = {}
        if self.dataset_config:
            load_kw["name"] = self.dataset_config

        def _apply_filter(dataset, filter_value):
            """Filter the HuggingFace dataset by an optionally specified column (e.g., "split") and value (e.g., "train")."""
            if self.filter_column and filter_value:
                return dataset.filter(lambda x: x[self.filter_column] == filter_value)  # may be slow if applied to multi-million-row datasets
            return dataset

        def _deduplicate_coco(dataset):
            """Deduplicate COCO datasets based on 'imgid' to keep only one caption per image."""
            if "COCO" in self.dataset_name.upper():
                img_ids = dataset["imgid"]
                seen_ids = set()
                unique_indices = []

                for idx, img_id in enumerate(img_ids):
                    if img_id not in seen_ids:
                        seen_ids.add(img_id)
                        unique_indices.append(idx)
                
                return dataset.select(unique_indices)
            return dataset

        if stage in (None, "fit"):
            raw = load_dataset(self.dataset_name, split=self.train_split, **load_kw)
            raw = _apply_filter(raw, self.train_filter_value)
            raw = _deduplicate_coco(raw)
            self.train_dataset = ImageTextDataset(
                raw, self.processor, self.image_column, self.caption_column, self.max_length
            )
        if stage in (None, "fit", "validate"):  # set up val_dataset in stage="fit" for in-training validation
            raw_val = load_dataset(self.dataset_name, split=self.val_split, **load_kw)
            raw_val = _apply_filter(raw_val, self.val_filter_value)
            raw_val = _deduplicate_coco(raw_val)
            self.val_dataset = ImageTextDataset(
                raw_val, self.processor, self.image_column, self.caption_column, self.max_length
            )
        if stage in (None, "test"):
            raw_test = load_dataset(self.dataset_name, split=self.test_split, **load_kw)
            raw_test = _apply_filter(raw_test, self.test_filter_value)
            raw_test = _deduplicate_coco(raw_test)
            self.test_dataset = ImageTextDataset(
                raw_test, self.processor, self.image_column, self.caption_column, self.max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
