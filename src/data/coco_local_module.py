"""In-training COCO-2017 retrieval validation from local files (offline-safe; one caption per image,
so ground truth is the diagonal). The reported multi-caption protocols live in src/eval/."""

from __future__ import annotations

import json
import os
from typing import Optional

import lightning as L
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.data.image_text_module import ImageTextDataset

DATA_DIR = os.environ.get("DATA_DIR", "data")   # override to point at your dataset root


class _CocoLocalRaw:
    """Indexable (image_path, caption) view returning {"image": PIL, "caption": str}."""

    def __init__(self, items: list[tuple[str, str]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        path, caption = self.items[idx]
        return {"image": Image.open(path).convert("RGB"), "caption": caption}


def _build_items(coco_dir: str, split: str, ann_name: str) -> list[tuple[str, str]]:
    """Return one (image_path, first_caption) pair per image."""
    ann_file = os.path.join(coco_dir, "annotations", ann_name)
    img_dir = os.path.join(coco_dir, split)
    with open(ann_file) as f:
        data = json.load(f)
    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    first_caption: dict[int, str] = {}
    for ann in data["annotations"]:
        first_caption.setdefault(ann["image_id"], ann["caption"])
    return [
        (os.path.join(img_dir, id_to_fname[iid]), cap)
        for iid, cap in first_caption.items()
        if iid in id_to_fname
    ]


class CocoLocalDataModule(L.LightningDataModule):
    """LightningDataModule for local COCO-2017 retrieval validation."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        coco_dir: str = os.path.join(DATA_DIR, "coco"),
        split: str = "val2017",
        ann_name: str = "captions_val2017.json",
        max_length: int = 77,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_images: Optional[int] = None,
        compute_retrieval_metrics: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.compute_retrieval_metrics = compute_retrieval_metrics

        self.coco_dir = coco_dir
        self.split = split
        self.ann_name = ann_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_images = max_images

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.val_dataset: Optional[ImageTextDataset] = None
        self.test_dataset: Optional[ImageTextDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        items = _build_items(self.coco_dir, self.split, self.ann_name)
        if self.max_images is not None:
            items = items[: self.max_images]
        ds = ImageTextDataset(
            _CocoLocalRaw(items), self.processor, "image", "caption", self.max_length
        )
        self.val_dataset = ds
        self.test_dataset = ds

    def _loader(self, ds: ImageTextDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(self.test_dataset)
