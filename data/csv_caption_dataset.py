#!/usr/bin/env python3
"""
CSV image-caption dataset for BiomedCLIP fine-tuning.
"""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CsvCaptionDataset(Dataset):
    """
    Dataset backed by a CSV with image path and caption columns.

    Expected default columns:
    - filename
    - Caption
    """
    def __init__(
        self,
        csv_path,
        processor,
        tokenizer,
        image_key="filename",
        caption_key="Caption",
        separator=",",
        image_root=None,
        max_length=77,
        max_samples=None,
    ):
        self.csv_path = Path(csv_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_key = image_key
        self.caption_key = caption_key
        self.separator = separator
        self.image_root = Path(image_root) if image_root else None
        self.max_length = max_length

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV does not exist: {self.csv_path}")

        self.samples = []
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=separator)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {self.csv_path}")
            missing = [key for key in (image_key, caption_key) if key not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"CSV {self.csv_path} missing columns {missing}. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row_idx, row in enumerate(reader, start=2):
                image_path = (row.get(image_key) or "").strip()
                caption = (row.get(caption_key) or "").strip()
                if not image_path or not caption:
                    continue
                resolved_path = self._resolve_image_path(image_path)
                if not resolved_path.exists():
                    raise FileNotFoundError(
                        f"Image path does not exist at CSV row {row_idx}: {resolved_path}"
                    )
                self.samples.append(
                    {
                        "image_path": resolved_path,
                        "caption": caption,
                    }
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No valid samples loaded from {self.csv_path}")

        print(
            f"[INFO] Loaded {len(self.samples)} image-caption pairs from {self.csv_path} "
            f"using image_key={image_key!r}, caption_key={caption_key!r}"
        )

    def _resolve_image_path(self, image_path):
        path = Path(image_path)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if callable(self.processor):
            processed = self.processor(image)
            if isinstance(processed, dict):
                pixel_values = processed["pixel_values"][0]
            else:
                pixel_values = processed
        else:
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        try:
            text_inputs = self.tokenizer(
                sample["caption"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            caption_ids = text_inputs["input_ids"][0]
            attention_mask = text_inputs["attention_mask"][0]
        except TypeError:
            tokenized = self.tokenizer([sample["caption"]])
            caption_ids = tokenized[0] if tokenized.dim() == 2 else tokenized
            attention_mask = torch.ones_like(caption_ids)

        return {
            "imgs": pixel_values,
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(caption_ids),
            "caption": sample["caption"],
            "image_path": str(sample["image_path"]),
        }
