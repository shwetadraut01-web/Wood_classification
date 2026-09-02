"""
Wood Species Dataset Preparation & Ingestion Pipeline
=====================================================
Orchestrates data ingestion for all 5 target wood species:
1. Oak      <- WOOD-AUTH (Category 4)
2. Pine     <- WOOD-AUTH (Category 8)
3. Walnut   <- WOOD-AUTH (Category 2)
4. Sheesham <- GOIMAI (Dalbergia sissoo)
5. Mahogany <- GOIMAI (Swietenia mahagoni, S. macrophylla, S. humilis)

Extracts raw images into:
data/wood_species/raw/<project_species>/

Generates comprehensive manifest:
data/wood_species/metadata/species_dataset_manifest.csv
"""

import csv
import os
import re
import sys
import zipfile
from collections import Counter
from typing import Dict, List, Optional
import cv2
import numpy as np

# Base Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPECIES_DIR = os.path.join(DATA_DIR, "wood_species")
RAW_DIR = os.path.join(SPECIES_DIR, "raw")
PROCESSED_DIR = os.path.join(SPECIES_DIR, "processed")
METADATA_DIR = os.path.join(SPECIES_DIR, "metadata")

# 5 Target Project Classes
PROJECT_CLASSES = ["oak", "pine", "walnut", "sheesham", "mahogany"]

# Source Mapping Definitions
WOOD_AUTH_CLASS_MAP = {
    4: "oak",      # Category 4 -> Oak (600 images)
    8: "pine",     # Category 8 -> Pine (332 images)
    2: "walnut",   # Category 2 -> Walnut (296 images)
}

GOIMAI_SOURCES = [
    {
        "zip_name": "Dalbergia sissoo.zip",
        "source_species": "Dalbergia sissoo",
        "project_species": "sheesham",
        "prefix": "goimai_dalbergia_sissoo_",
    },
    {
        "zip_name": "Swietenia mahagoni.zip",
        "source_species": "Swietenia mahagoni",
        "project_species": "mahogany",
        "prefix": "goimai_swietenia_mahagoni_",
    },
    {
        "zip_name": "Swietenia macrophylla.zip",
        "source_species": "Swietenia macrophylla",
        "project_species": "mahogany",
        "prefix": "goimai_swietenia_macrophylla_",
    },
    {
        "zip_name": "Swietenia humilis.zip",
        "source_species": "Swietenia humilis",
        "project_species": "mahogany",
        "prefix": "goimai_swietenia_humilis_",
    },
]


def check_source_availability() -> Dict[str, bool]:
    """
    Checks the local presence of all required source dataset archives.
    """
    wood_auth_zip = os.path.join(DATA_DIR, "Wood Dataset.zip")
    status = {
        "WOOD-AUTH (Wood Dataset.zip)": os.path.exists(wood_auth_zip),
    }
    for src in GOIMAI_SOURCES:
        path = os.path.join(DATA_DIR, src["zip_name"])
        status[f"GOIMAI - {src['source_species']} ({src['project_species'].capitalize()})"] = os.path.exists(path)
    return status


def ingest_wood_auth_dataset(
    zip_path: Optional[str] = None,
    raw_dir: str = RAW_DIR,
) -> List[Dict]:
    """
    Extracts Oak, Pine, and Walnut from WOOD-AUTH into raw/ folders and returns manifest records.
    """
    if zip_path is None:
        zip_path = os.path.join(DATA_DIR, "Wood Dataset.zip")

    print("\n" + "=" * 75)
    print(" INGESTING WOOD-AUTH (Oak, Pine, Walnut) ")
    print("=" * 75)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Source archive not found: {zip_path}")

    manifest_rows = []
    class_counts = Counter()
    pattern = re.compile(r"^(\d+)_(\d+)\.jpe?g$", re.IGNORECASE)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.infolist():
            if entry.is_dir() or entry.filename.endswith("/"):
                continue

            basename = os.path.basename(entry.filename)
            match = pattern.match(basename)
            if not match:
                continue

            category_id = int(match.group(2))
            if category_id not in WOOD_AUTH_CLASS_MAP:
                continue

            species = WOOD_AUTH_CLASS_MAP[category_id]
            source_split = "train" if "Train" in entry.filename else "test"
            dest_filename = f"{source_split}_{basename}"
            dest_filepath = os.path.join(raw_dir, species, dest_filename)

            raw_bytes = zf.read(entry)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                print(f" [ERROR] Could not decode {entry.filename}")
                continue

            with open(dest_filepath, "wb") as f_out:
                f_out.write(raw_bytes)

            class_counts[species] += 1
            manifest_rows.append({
                "source_dataset": "WOOD-AUTH",
                "source_species": f"Category {category_id} ({species.capitalize()})",
                "project_species": species,
                "original_filename": basename,
                "destination_filename": dest_filename,
                "destination_path": os.path.relpath(dest_filepath, PROJECT_ROOT),
                "image_width": img.shape[1],
                "image_height": img.shape[0],
                "channels": img.shape[2],
                "file_size_bytes": len(raw_bytes),
            })

    for sp in ["oak", "pine", "walnut"]:
        print(f"  - {sp.capitalize():<10}: {class_counts[sp]:4d} images ingested")
    return manifest_rows


def ingest_goimai_dataset(raw_dir: str = RAW_DIR) -> List[Dict]:
    """
    Extracts Sheesham and Mahogany (3 Swietenia species) from GOIMAI ZIPs into raw/ folders.
    """
    print("\n" + "=" * 75)
    print(" INGESTING GOIMAI (Sheesham & Mahogany) ")
    print("=" * 75)

    manifest_rows = []
    source_counts = Counter()

    for src_config in GOIMAI_SOURCES:
        zip_path = os.path.join(DATA_DIR, src_config["zip_name"])
        source_species = src_config["source_species"]
        project_species = src_config["project_species"]
        prefix = src_config["prefix"]

        if not os.path.exists(zip_path):
            print(f" [ERROR] Missing ZIP file: {zip_path}")
            continue

        dest_dir = os.path.join(raw_dir, project_species)
        os.makedirs(dest_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = [e for e in zf.infolist() if not e.is_dir() and not e.filename.endswith("/")]
            print(f" Processing {source_species:<25} ({len(entries)} images in {src_config['zip_name']})...")

            for entry in entries:
                basename = os.path.basename(entry.filename)
                _, ext = os.path.splitext(basename.lower())
                if ext not in [".jpg", ".jpeg"]:
                    continue

                dest_filename = f"{prefix}{basename}"
                dest_filepath = os.path.join(dest_dir, dest_filename)

                raw_bytes = zf.read(entry)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    print(f" [ERROR] Could not decode {entry.filename}")
                    continue

                with open(dest_filepath, "wb") as f_out:
                    f_out.write(raw_bytes)

                source_counts[source_species] += 1
                manifest_rows.append({
                    "source_dataset": "GOIMAI",
                    "source_species": source_species,
                    "project_species": project_species,
                    "original_filename": basename,
                    "destination_filename": dest_filename,
                    "destination_path": os.path.relpath(dest_filepath, PROJECT_ROOT),
                    "image_width": img.shape[1],
                    "image_height": img.shape[0],
                    "channels": img.shape[2],
                    "file_size_bytes": len(raw_bytes),
                })

    print("\n GOIMAI Source Ingestion Breakdown:")
    for src in GOIMAI_SOURCES:
        s_name = src["source_species"]
        p_name = src["project_species"]
        print(f"  - {s_name:<25} -> Class '{p_name}': {source_counts[s_name]:4d} images")

    return manifest_rows


def run_full_preparation():
    print("=" * 75)
    print(" FULL 5-CLASS WOOD SPECIES DATASET INGESTION ")
    print("=" * 75)

    os.makedirs(METADATA_DIR, exist_ok=True)
    for cls in PROJECT_CLASSES:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)

    # 1. Ingest WOOD-AUTH
    wood_auth_rows = ingest_wood_auth_dataset()

    # 2. Ingest GOIMAI
    goimai_rows = ingest_goimai_dataset()

    # 3. Combine Manifest
    all_manifest_rows = wood_auth_rows + goimai_rows
    manifest_csv_path = os.path.join(METADATA_DIR, "species_dataset_manifest.csv")

    fieldnames = [
        "source_dataset",
        "source_species",
        "project_species",
        "original_filename",
        "destination_filename",
        "destination_path",
        "image_width",
        "image_height",
        "channels",
        "file_size_bytes",
    ]

    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    print("\n" + "=" * 75)
    print(" COMBINED MANIFEST SUMMARY")
    print("=" * 75)
    print(f" Total Records Written : {len(all_manifest_rows)}")
    print(f" Manifest Location     : {manifest_csv_path}")

    class_totals = Counter(r["project_species"] for r in all_manifest_rows)
    print("\n Final 5-Class Distribution:")
    for cls in PROJECT_CLASSES:
        print(f"  - {cls.capitalize():<12}: {class_totals[cls]:4d} images")
    print(f"  {'-'*25}")
    print(f"  {'TOTAL':<12}: {sum(class_totals.values()):4d} images")
    print("=" * 75)


if __name__ == "__main__":
    run_full_preparation()
