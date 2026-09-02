"""
Test & Verification Script for LBP Feature Extraction (Module 1)
================================================================
This script tests:
1. Loading and preprocessing a sample wood texture image.
2. Extracting Local Binary Pattern (LBP) features (P=8, R=1, method='uniform').
3. Displaying formatted tables of normalized histogram bins and statistical moments.
4. Validating mathematical bounds and probability constraints:
   - Histogram values >= 0
   - Histogram values <= 1
   - Histogram sum ~= 1.0
   - Histogram mean is valid (> 0)
   - Histogram standard deviation >= 0
   - Histogram entropy >= 0
"""

import os
import sys
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image
from src.lbp_features import extract_lbp_features, compute_lbp_map


def test_lbp_extraction():
    print("=" * 70)
    print(" Module 1: Wood Texture Analysis - LBP Feature Extraction Test ")
    print("=" * 70)

    sample_image_path = os.path.join(PROJECT_ROOT, "data", "sample_images", "sample_wood.png")

    if not os.path.exists(sample_image_path):
        from tests.test_preprocessing import generate_sample_wood_image
        print(f"Creating sample wood image at: {sample_image_path}")
        generate_sample_wood_image(sample_image_path)

    # 1. Preprocess the image
    print(f"\n[Step 1] Preprocessing input image: {sample_image_path}")
    preprocessed_gray = preprocess_image(sample_image_path, target_size=(512, 512), denoise=True)
    print(f"  - Preprocessed image shape : {preprocessed_gray.shape}")
    print(f"  - Data type                : {preprocessed_gray.dtype}")
    print(f"  - Intensity range          : [{preprocessed_gray.min()}, {preprocessed_gray.max()}]")

    # 2. Extract LBP features
    print(f"\n[Step 2] Extracting Uniform LBP features (Radius=1, Points=8)...")
    lbp_results = extract_lbp_features(
        preprocessed_gray,
        radius=1,
        points=8,
        method="uniform",
    )

    # 3. Print Extracted Configuration & Statistical Summary
    print("\n" + "-" * 70)
    print(" LBP CONFIGURATION & STATISTICAL MOMENTS")
    print("-" * 70)
    print(f"  Neighborhood Radius (R)       : {lbp_results['radius']} px")
    print(f"  Sampling Points (P)           : {lbp_results['number_of_points']}")
    print(f"  Method                        : '{lbp_results['method']}'")
    print(f"  Total Histogram Bins          : {lbp_results['num_bins']}")
    print(f"  Histogram Mean                : {lbp_results['histogram_mean']:.4f}")
    print(f"  Histogram Std Deviation       : {lbp_results['histogram_std']:.4f}")
    print(f"  Histogram Shannon Entropy     : {lbp_results['histogram_entropy']:.4f} bits")
    print(f"  Histogram Energy (Uniformity) : {lbp_results['histogram_energy']:.4f}")
    print(f"  Uniform Pattern Ratio         : {lbp_results['uniform_ratio'] * 100:.2f}%")
    print(f"  Non-Uniform Noise Ratio       : {lbp_results['non_uniform_ratio'] * 100:.2f}%")
    print("-" * 70)

    # 4. Print Normalized Histogram Probability Table
    hist = lbp_results["histogram"]
    print("\n" + "-" * 70)
    print(" NORMALIZED LBP HISTOGRAM PROBABILITIES (P=8, R=1, 'uniform')")
    print("-" * 70)
    print(f" {'Bin #':<8} | {'Pattern Description':<32} | {'Probability':<14} | {'Visual Bar'}")
    print("-" * 70)
    
    bin_labels = [
        "Bin 0 (Flat / Dark Region)",
        "Bin 1 (Edge / Transition 1)",
        "Bin 2 (Edge / Transition 2)",
        "Bin 3 (Curved Edge / Corner 1)",
        "Bin 4 (Curved Edge / Corner 2)",
        "Bin 5 (Curved Edge / Corner 3)",
        "Bin 6 (Edge / Transition 3)",
        "Bin 7 (Edge / Transition 4)",
        "Bin 8 (Flat / Bright Region)",
        "Bin 9 (Non-Uniform / Noise)",
    ]

    for i, prob in enumerate(hist):
        label = bin_labels[i] if i < len(bin_labels) else f"Bin {i}"
        bar = "█" * int(prob * 50)
        print(f" {i:<8} | {label:<32} | {prob:<14.4f} | {bar}")
    print("-" * 70)
    print(f" {'TOTAL':<8} | {'Sum of Probabilities':<32} | {sum(hist):<14.4f} |")
    print("-" * 70)

    # 5. Assertions & Mathematical Bound Verification
    print("\n" + "=" * 70)
    print(" VERIFYING MATHEMATICAL BOUNDS & DATA INTEGRITY...")
    print("=" * 70)

    # Histogram exists and is non-empty
    assert "histogram" in lbp_results and len(lbp_results["histogram"]) > 0, "Histogram missing or empty!"
    print(f" [PASS] Histogram exists with {len(lbp_results['histogram'])} bins")

    # Key validation
    assert "points" in lbp_results and lbp_results["points"] == 8, "Key 'points' missing or invalid!"
    assert "radius" in lbp_results and lbp_results["radius"] == 1, "Key 'radius' missing or invalid!"
    assert "method" in lbp_results and lbp_results["method"] == "uniform", "Key 'method' missing or invalid!"
    print(" [PASS] Required keys ('radius', 'points', 'method') present and valid")

    # All histogram probabilities >= 0 and <= 1
    assert all(p >= 0.0 for p in hist), "Negative histogram value found!"
    assert all(p <= 1.0 for p in hist), "Histogram value > 1.0 found!"
    print(" [PASS] All histogram values are bounded in [0.0, 1.0]")

    # Histogram sum ~= 1.0
    hist_sum = sum(hist)
    assert np.isclose(hist_sum, 1.0, atol=1e-5), f"Histogram sum is not 1.0! Got {hist_sum}"
    print(f" [PASS] Histogram sum == 1.0 (Exact: {hist_sum:.6f})")

    # Histogram mean > 0
    assert lbp_results["histogram_mean"] > 0.0, "Histogram mean must be positive"
    print(f" [PASS] Histogram mean is valid (> 0.0: {lbp_results['histogram_mean']:.4f})")

    # Histogram standard deviation >= 0
    assert lbp_results["histogram_std"] >= 0.0, "Histogram std must be >= 0.0"
    print(f" [PASS] Histogram std dev is valid (>= 0.0: {lbp_results['histogram_std']:.4f})")

    # Histogram entropy >= 0
    assert lbp_results["histogram_entropy"] >= 0.0, "Histogram entropy must be >= 0.0"
    print(f" [PASS] Histogram Shannon entropy is valid (>= 0.0: {lbp_results['histogram_entropy']:.4f} bits)")

    # Check for NaN or Infinite values across all numeric fields
    for k, v in lbp_results.items():
        if isinstance(v, (int, float)):
            assert not np.isnan(v), f"Feature '{k}' is NaN!"
            assert not np.isinf(v), f"Feature '{k}' is Infinite!"
        elif isinstance(v, list):
            assert not any(np.isnan(x) for x in v), f"List '{k}' contains NaN!"
            assert not any(np.isinf(x) for x in v), f"List '{k}' contains Infinite!"
    print(" [PASS] No returned numeric feature is NaN or Infinite")

    # Spatial LBP map dimensions
    lbp_map = compute_lbp_map(preprocessed_gray, radius=1, n_points=8, method="uniform")
    assert lbp_map.shape == preprocessed_gray.shape, "LBP map shape mismatch!"
    print(f" [PASS] 2D LBP texture map matches preprocessed image shape: {lbp_map.shape}")

    print("\n" + "=" * 70)
    print(" ALL LBP FEATURE EXTRACTION CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 70)


if __name__ == "__main__":
    test_lbp_extraction()
