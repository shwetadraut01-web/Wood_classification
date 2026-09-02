"""
Wood Species Dataset Stratified Splitting Script
================================================
Creates a reproducible, stratified 70/15/15 project-level split:
- 70% Train (~1,154 images)
- 15% Validation (~247 images)
- 15% Test (~247 images)
- Fixed Random Seed: 42

Copies files from data/wood_species/raw/<class>/ into:
data/wood_species/splits/{train, validation, test}/<class>/

Generates:
data/wood_species/metadata/split_manifest.csv
"""

import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Base Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPECIES_DIR = os.path.join(DATA_DIR, "wood_species")
RAW_DIR = os.path.join(SPECIES_DIR, "raw")
SPLITS_DIR = os.path.join(SPECIES_DIR, "splits")
METADATA_DIR = os.path.join(SPECIES_DIR, "metadata")

PROJECT_CLASSES = ["oak", "pine", "walnut", "sheesham", "mahogany"]
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42


def calculate_split_counts(total: int) -> Tuple[int, int, int]:
    """
    Computes exact integer counts for (train, val, test) ensuring the sum equals total.
    """
    n_train = int(round(total * SPLIT_RATIOS["train"]))
    n_val = int(round(total * SPLIT_RATIOS["validation"]))
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def create_stratified_splits(
    raw_dir: str = RAW_DIR,
    splits_dir: str = SPLITS_DIR,
    metadata_dir: str = METADATA_DIR,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Performs stratified 70/15/15 dataset splitting across the 5 project classes.
    """
    print("=" * 78)
    print(" 5-CLASS WOOD SPECIES DATASET STRATIFIED SPLITTING ")
    print("=" * 78)
    print(f" Source Directory    : {raw_dir}")
    print(f" Target Splits Root  : {splits_dir}")
    print(f" Random Seed         : {seed}")
    print(f" Split Ratios        : 70% Train | 15% Validation | 15% Test")

    # Load existing manifest for rich metadata if available
    raw_manifest_path = os.path.join(metadata_dir, "species_dataset_manifest.csv")
    raw_metadata_lookup = {}
    if os.path.exists(raw_manifest_path):
        with open(raw_manifest_path, "r", encoding="utf-8") as f_m:
            reader = csv.DictReader(f_m)
            for row in reader:
                dest_fn = row["destination_filename"]
                raw_metadata_lookup[dest_fn] = row

    # Create destination directories
    split_names = ["train", "validation", "test"]
    for sp_name in split_names:
        for cls in PROJECT_CLASSES:
            os.makedirs(os.path.join(splits_dir, sp_name, cls), exist_ok=True)

    manifest_rows = []
    class_split_counts = defaultdict(lambda: Counter())
    total_split_counts = Counter()

    for cls in PROJECT_CLASSES:
        cls_raw_dir = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_raw_dir):
            raise FileNotFoundError(f"Raw directory missing for class: {cls}")

        filenames = sorted([
            f for f in os.listdir(cls_raw_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
        ])
        total_cls_images = len(filenames)
        if total_cls_images == 0:
            raise ValueError(f"No raw images found in {cls_raw_dir}")

        # Deterministic shuffle with fixed seed
        rng = random.Random(seed + sum(ord(c) for c in cls))
        shuffled_files = filenames.copy()
        rng.shuffle(shuffled_files)

        n_train, n_val, n_test = calculate_split_counts(total_cls_images)
        assert n_train + n_val + n_test == total_cls_images, "Split sum mismatch!"

        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train:n_train + n_val]
        test_files = shuffled_files[n_train + n_val:]

        splits_map = {
            "train": train_files,
            "validation": val_files,
            "test": test_files,
        }

        for sp_name, files in splits_map.items():
            for fn in files:
                src_path = os.path.join(cls_raw_dir, fn)
                dst_path = os.path.join(splits_dir, sp_name, cls, fn)

                # Exact copy without modifying original or re-compressing
                shutil.copy2(src_path, dst_path)

                # Lookup metadata or fallback
                meta = raw_metadata_lookup.get(fn, {})
                source_ds = meta.get("source_dataset", "WOOD-AUTH" if cls in ["oak", "pine", "walnut"] else "GOIMAI")
                source_sp = meta.get("source_species", cls.capitalize())
                orig_fn = meta.get("original_filename", fn)
                w = int(meta.get("image_width", 0))
                h = int(meta.get("image_height", 0))
                ch = int(meta.get("channels", 3))

                manifest_rows.append({
                    "source_dataset": source_ds,
                    "source_species": source_sp,
                    "project_species": cls,
                    "original_filename": orig_fn,
                    "source_path": os.path.relpath(src_path, PROJECT_ROOT),
                    "split": sp_name,
                    "destination_path": os.path.relpath(dst_path, PROJECT_ROOT),
                    "image_width": w,
                    "image_height": h,
                    "channels": ch,
                    "random_seed": seed,
                })

                class_split_counts[cls][sp_name] += 1
                total_split_counts[sp_name] += 1

    # Write Split Manifest CSV
    split_manifest_path = os.path.join(metadata_dir, "split_manifest.csv")
    fieldnames = [
        "source_dataset",
        "source_species",
        "project_species",
        "original_filename",
        "source_path",
        "split",
        "destination_path",
        "image_width",
        "image_height",
        "channels",
        "random_seed",
    ]

    with open(split_manifest_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "-" * 78)
    print(" STRATIFIED SPLIT DISTRIBUTION TABLE")
    print("-" * 78)
    print(f" {'Class Name':<12} | {'Total':<8} | {'Train (70%)':<12} | {'Val (15%)':<11} | {'Test (15%)':<11}")
    print(" " + "-" * 78)
    for cls in PROJECT_CLASSES:
        tot = sum(class_split_counts[cls].values())
        tr = class_split_counts[cls]["train"]
        va = class_split_counts[cls]["validation"]
        te = class_split_counts[cls]["test"]
        print(f" {cls.capitalize():<12} | {tot:<8} | {tr:<12} | {va:<11} | {te:<11}")
    print(" " + "-" * 78)
    print(f" {'TOTAL':<12} | {len(manifest_rows):<8} | {total_split_counts['train']:<12} | {total_split_counts['validation']:<11} | {total_split_counts['test']:<11}")
    print(" " + "-" * 78)

    print(f"\n Manifest generated at: {split_manifest_path} ({len(manifest_rows)} records)")
    return {
        "total_records": len(manifest_rows),
        "manifest_path": split_manifest_path,
        "class_counts": dict(class_split_counts),
        "total_counts": dict(total_split_counts),
    }


if __name__ == "__main__":
    create_stratified_splits()
