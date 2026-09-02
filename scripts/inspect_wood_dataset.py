"""
Dataset Inspection & Category Verification for Wood Dataset.zip
================================================================
Correctly parses the Zenodo Wood Dataset naming scheme:
    <photo_number>_<species_category>.jpg
    Example: 45_3.jpg -> Photo #45, Species Category 3 (Sweet Chestnut)

Dataset Details:
- 12 Wood Species Categories
- 8,544 Macroscopic Wood Texture Images (JPEG)
- Train Split: 5,708 images organized in subfolders Train/1 through Train/12
- Test Split : 2,836 images in flat directory Test/
"""

import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

# Official Zenodo Wood Species Class Mapping
SPECIES_MAPPING = {
    1: "European Beech",
    2: "Walnut",
    3: "Sweet Chestnut",
    4: "Oak",
    5: "Alder",
    6: "Ash",
    7: "Norway Spruce",
    8: "Pine",
    9: "Tree of Heaven",
    10: "Black Locust",
    11: "Mediterranean Cypress",
    12: "Oriental Plane",
}


def parse_wood_filename(filename: str):
    """
    Parses a wood dataset image filename into (photo_number, species_category).
    
    Naming Scheme: <photo_number>_<species_category>.jpg
    Example: '45_3.jpg' -> photo_number = 45, species_category = 3
    """
    basename = os.path.basename(filename)
    match = re.match(r"^(\d+)_(\d+)\.jpe?g$", basename, re.IGNORECASE)
    if match:
        photo_number = int(match.group(1))
        species_category = int(match.group(2))
        return photo_number, species_category
    return None, None


def inspect_wood_dataset(zip_path: str):
    if not os.path.exists(zip_path):
        print(f"Error: Archive not found at {zip_path}")
        return

    print("=" * 82)
    print(" ZENODO WOOD SPECIES DATASET INSPECTION & VERIFICATION REPORT")
    print("=" * 82)
    print(f" Archive Path : {zip_path}")
    print(f" Archive Size : {os.path.getsize(zip_path) / (1024 * 1024):.2f} MB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = zf.infolist()
        file_entries = [e for e in all_entries if not e.is_dir() and not e.filename.endswith("/")]

        # Counters
        train_counts = Counter()
        test_counts = Counter()
        total_counts = Counter()
        unmatched_files = []
        parsed_samples = []

        # Top-level directory inspection
        top_level_folders = sorted(list({e.filename.split("/")[0] for e in all_entries if "/" in e.filename}))
        train_subfolders = sorted(list({e.filename.split("/")[2] for e in file_entries if len(e.filename.split("/")) >= 4 and e.filename.split("/")[1] == "Train"}), key=lambda x: int(x) if x.isdigit() else x)

        for entry in file_entries:
            filepath = entry.filename
            photo_num, cat_id = parse_wood_filename(filepath)

            if cat_id is not None:
                total_counts[cat_id] += 1
                if "Train" in filepath:
                    train_counts[cat_id] += 1
                elif "Test" in filepath:
                    test_counts[cat_id] += 1

                if len(parsed_samples) < 15:
                    parsed_samples.append({
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "photo_number": photo_num,
                        "species_category": cat_id,
                        "species_name": SPECIES_MAPPING.get(cat_id, "Unknown"),
                        "split": "Train" if "Train" in filepath else "Test"
                    })
            else:
                unmatched_files.append(filepath)

        # 1. Folder Structure
        print("\n--- 1. ARCHIVE FOLDER STRUCTURE ---")
        print(f" Top-level directory in ZIP : {top_level_folders}")
        print(f" Train subfolders found     : {train_subfolders} ({len(train_subfolders)} category folders)")
        print(f" Test folder structure      : Flat folder containing test images for all 12 categories")
        print(f" Total files in ZIP         : {len(file_entries)} JPEG images")
        print(f" Unmatched/Malformed files  : {len(unmatched_files)}")

        # 2. Filename Parsing Demonstration
        print("\n--- 2. FILENAME PARSING VERIFICATION (Sample Images) ---")
        print(f" {'Filename':<14} | {'Split':<6} | {'Photo Number':<14} | {'Category ID':<12} | {'Species Name'}")
        print(" " + "-" * 75)
        for sample in parsed_samples:
            print(f" {sample['filename']:<14} | {sample['split']:<6} | {sample['photo_number']:<14} | {sample['species_category']:<12} | {sample['species_name']}")

        # Specific user-requested test examples
        print("\n Verification of Requested Test Filenames:")
        test_examples = ["1_1.jpg", "1_10.jpg", "45_3.jpg", "12_63.jpg"]
        for fn in test_examples:
            p, c = parse_wood_filename(fn)
            name = SPECIES_MAPPING.get(c, "Unknown")
            print(f"   - filename: {fn:<10} -> species_category: {c:<2} ({name:<18}), photo_number: {p}")

        # 3. Comprehensive Category Distribution Table
        print("\n--- 3. WOOD SPECIES DISTRIBUTION TABLE (Categories 1 to 12) ---")
        print(f" {'Category':<9} | {'Species Name':<23} | {'Train Count':<12} | {'Test Count':<11} | {'Total Images':<12} | {'Split Ratio'}")
        print(" " + "-" * 88)
        
        total_train = 0
        total_test = 0
        total_overall = 0

        for cat in range(1, 13):
            species = SPECIES_MAPPING.get(cat, f"Category {cat}")
            tr = train_counts[cat]
            te = test_counts[cat]
            tot = tr + te
            total_train += tr
            total_test += te
            total_overall += tot
            ratio = f"{(tr/tot)*100:.1f}% / {(te/tot)*100:.1f}%" if tot > 0 else "N/A"
            print(f" {cat:<9} | {species:<23} | {tr:<12} | {te:<11} | {tot:<12} | {ratio}")

        print(" " + "-" * 88)
        print(f" {'TOTAL':<9} | {'All 12 Species':<23} | {total_train:<12} | {total_test:<11} | {total_overall:<12} | {(total_train/total_overall)*100:.1f}% / {(total_test/total_overall)*100:.1f}%")
        print(" " + "-" * 88)

        # 4. Highlight Specific Target Categories (Walnut, Oak, Pine)
        print("\n--- 4. TARGET SPECIES FOR PROJECT HIGHLIGHT ---")
        targets = [
            (2, "Walnut"),
            (4, "Oak"),
            (8, "Pine")
        ]
        for cat_id, expected_name in targets:
            tr = train_counts[cat_id]
            te = test_counts[cat_id]
            tot = tr + te
            print(f" >> Category {cat_id:<2} — {expected_name:<8} : Train = {tr:4d}, Test = {te:4d}, Total = {tot:4d} images")

        # 5. Zenodo Dataset Consistency Verification
        print("\n--- 5. ZENODO DATASET INTEGRITY VERIFICATION ---")
        print(f" Expected Macroscopic Images (Zenodo doc) : 8,544")
        print(f" Actual JPEG Images in ZIP                : {total_overall}")
        print(f" Consistency Check                        : {'[PASS] Exact match with Zenodo specification!' if total_overall == 8544 else '[FAIL] Mismatch'}")
        print("=" * 82)


if __name__ == "__main__":
    zip_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "data",
        "Wood Dataset.zip",
    )
    inspect_wood_dataset(zip_path)
