"""
Wood Species Dataset Health & Integrity Checker
================================================
Inspects data/wood_species/ and reports:
- Number of images in each of the 5 classes (oak, pine, walnut, sheesham, mahogany)
- File format extensions
- Image dimensions, aspect ratios, and color channels
- Unreadable or corrupted files
- Duplicate destination filenames
- Overall total image counts in raw/ and processed/ directories
"""

import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

# Base Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPECIES_DIR = os.path.join(DATA_DIR, "wood_species")

PROJECT_CLASSES = ["oak", "pine", "walnut", "sheesham", "mahogany"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def inspect_directory_split(split_name: str, split_path: str) -> Dict:
    """
    Scans a dataset directory split (e.g. raw or processed) across the 5 project classes.
    """
    print("\n" + "-" * 82)
    print(f" INSPECTING SPLIT: {split_name.upper()} ({split_path})")
    print("-" * 82)

    if not os.path.exists(split_path):
        print(f" [INFO] Directory '{split_path}' does not exist.")
        return {"total": 0, "class_counts": {}}

    class_counts = Counter()
    class_extensions = defaultdict(Counter)
    dimensions_found = defaultdict(set)
    corrupted_files = []
    duplicate_tracker = defaultdict(list)
    total_images = 0

    for cls in PROJECT_CLASSES:
        cls_folder = os.path.join(split_path, cls)
        if not os.path.exists(cls_folder):
            class_counts[cls] = 0
            continue

        entries = os.listdir(cls_folder)
        image_files = [f for f in entries if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS and not f.startswith(".")]

        class_counts[cls] = len(image_files)
        total_images += len(image_files)

        for img_name in image_files:
            img_path = os.path.join(cls_folder, img_name)
            ext = os.path.splitext(img_name.lower())[1]
            class_extensions[cls][ext] += 1
            duplicate_tracker[img_name].append(cls)

            # Check image integrity & dimensions
            try:
                img = cv2.imread(img_path)
                if img is None:
                    corrupted_files.append(img_path)
                else:
                    dimensions_found[cls].add((img.shape[0], img.shape[1], img.shape[2]))
            except Exception as e:
                corrupted_files.append(f"{img_path} (Error: {e})")

    # Display class count table
    print(f" {'Class Name':<12} | {'Image Count':<12} | {'Status':<14} | {'Extensions':<10} | {'Dimensions (H, W, C)'}")
    print(" " + "-" * 85)
    for cls in PROJECT_CLASSES:
        cnt = class_counts[cls]
        status = "Populated" if cnt > 0 else "Empty"
        exts = ", ".join(class_extensions[cls].keys()) if class_extensions[cls] else "None"
        
        dims_list = sorted(list(dimensions_found[cls]))
        if len(dims_list) == 1:
            dims_str = f"{dims_list[0][0]}x{dims_list[0][1]} ({dims_list[0][2]} ch)"
        elif len(dims_list) > 1:
            dims_str = f"{len(dims_list)} variants (e.g. {dims_list[0][0]}x{dims_list[0][1]} to {dims_list[-1][0]}x{dims_list[-1][1]})"
        else:
            dims_str = "N/A"

        print(f" {cls.capitalize():<12} | {cnt:<12} | {status:<14} | {exts:<10} | {dims_str}")
    print(" " + "-" * 85)
    print(f" {'TOTAL':<12} | {total_images:<12} |")

    # Report duplicates and corruption
    duplicates = {name: classes for name, classes in duplicate_tracker.items() if len(classes) > 1}
    if duplicates:
        print(f"\n [WARNING] Found {len(duplicates)} duplicate filenames across classes:")
        for name, classes in list(duplicates.items())[:5]:
            print(f"   - '{name}' in classes: {classes}")
    else:
        print(" [CHECK] No duplicate filenames detected across classes.")

    if corrupted_files:
        print(f"\n [ERROR] Found {len(corrupted_files)} unreadable/corrupt images:")
        for cf in corrupted_files[:5]:
            print(f"   - {cf}")
    else:
        print(" [CHECK] 100% of images are intact and decoded with zero corruption.")

    return {
        "total": total_images,
        "class_counts": dict(class_counts),
        "dimensions": {cls: list(dimensions_found[cls]) for cls in PROJECT_CLASSES},
        "corrupted_count": len(corrupted_files),
        "duplicates_count": len(duplicates),
    }


def run_dataset_check():
    print("=" * 82)
    print(" WOOD SPECIES DATASET HEALTH CHECK (data/wood_species/) ")
    print("=" * 82)
    print(f" Target Project Classes : {', '.join([c.capitalize() for c in PROJECT_CLASSES])} (5 classes)")
    print(f" Dataset Root Directory : {SPECIES_DIR}")

    raw_stats = inspect_directory_split("raw", os.path.join(SPECIES_DIR, "raw"))
    processed_stats = inspect_directory_split("processed", os.path.join(SPECIES_DIR, "processed"))

    print("\n" + "=" * 82)
    print(" OVERALL DATASET SUMMARY")
    print("=" * 82)
    print(f" Total Raw Images        : {raw_stats['total']} images")
    print(f" Total Processed Images  : {processed_stats['total']} images")
    print("=" * 82)


if __name__ == "__main__":
    run_dataset_check()
