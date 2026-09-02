"""
Test & Verification Script for GLCM Feature Extraction (Module 1)
=================================================================
This script tests:
1. Loading and preprocessing a sample wood texture image.
2. Extracting GLCM features across angles (0°, 45°, 90°, 135°) and distances (1, 3, 5 px).
3. Displaying formatted tables of directional values and aggregated summary metrics.
4. Validating mathematical bounds:
   - Contrast >= 0
   - Correlation in [-1, 1]
   - Energy in [0, 1]
   - Homogeneity in [0, 1]
"""

import os
import sys
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image
from src.glcm_features import extract_glcm_features


def test_glcm_extraction():
    print("=" * 70)
    print(" Module 1: Wood Texture Analysis - GLCM Feature Extraction Test ")
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

    # 2. Extract GLCM features
    print(f"\n[Step 2] Extracting GLCM features...")
    distances = [1, 3, 5]
    angles_deg = [0, 45, 90, 135]
    angles_rad = [np.radians(deg) for deg in angles_deg]

    glcm_results = extract_glcm_features(
        preprocessed_gray,
        distances=distances,
        angles=angles_rad,
        angles_deg=angles_deg,
        levels=256,
    )

    summary = glcm_results["summary"]
    details = glcm_results["details"]
    config = glcm_results["config"]

    # 3. Print Aggregated Summary Table
    print("\n" + "-" * 70)
    print(" GLCM AGGREGATED SUMMARY (Averaged across all distances & angles)")
    print("-" * 70)
    print(f" {'Feature':<18} | {'Mean Value':<15} | {'Std Deviation':<15}")
    print("-" * 70)
    print(f" {'Contrast':<18} | {summary['contrast_mean']:<15.4f} | {summary['contrast_std']:<15.4f}")
    print(f" {'Correlation':<18} | {summary['correlation_mean']:<15.4f} | {summary['correlation_std']:<15.4f}")
    print(f" {'Energy':<18} | {summary['energy_mean']:<15.4f} | {summary['energy_std']:<15.4f}")
    print(f" {'Homogeneity':<18} | {summary['homogeneity_mean']:<15.4f} | {summary['homogeneity_std']:<15.4f}")
    print("-" * 70)

    # 4. Print Directional & Distance Breakdown
    print("\n" + "-" * 70)
    print(" DIRECTIONAL & DISTANCE BREAKDOWN (Per Angle & Distance)")
    print("-" * 70)
    for feature_name in ["contrast", "correlation", "energy", "homogeneity"]:
        feat = details[feature_name]
        print(f"\n>> {feature_name.upper()}:")
        
        # Breakdown by Distance
        dist_str = " | ".join([f"{k}: {v:.4f}" for k, v in feat["by_distance"].items()])
        print(f"   By Distance -> {dist_str}")
        
        # Breakdown by Angle
        angle_str = " | ".join([f"{k}: {v:.4f}" for k, v in feat["by_angle"].items()])
        print(f"   By Angle    -> {angle_str}")

        # Raw Matrix (Distance x Angle)
        matrix = np.array(feat["raw_matrix"])
        print(f"   Matrix [Distance x Angle]:")
        for d_idx, d in enumerate(distances):
            row_str = "  ".join([f"{matrix[d_idx, a_idx]:8.4f}" for a_idx in range(len(angles_deg))])
            print(f"     d={d}px: [ {row_str} ]")

    # 5. Assertions & Mathematical Bound Verification
    print("\n" + "=" * 70)
    print(" VERIFYING MATHEMATICAL BOUNDS & DATA INTEGRITY...")
    print("=" * 70)

    # Contrast must be >= 0
    assert summary["contrast_mean"] >= 0.0, f"Invalid Contrast: {summary['contrast_mean']}"
    assert all(c >= 0.0 for c in np.array(details["contrast"]["raw_matrix"]).flatten()), "Negative contrast found!"
    print(" [PASS] Contrast >= 0 (measures wood grain intensity variation)")

    # Correlation must be within [-1, 1]
    assert -1.0 <= summary["correlation_mean"] <= 1.0, f"Invalid Correlation: {summary['correlation_mean']}"
    assert all(-1.0 <= c <= 1.0 for c in np.array(details["correlation"]["raw_matrix"]).flatten()), "Correlation out of [-1, 1]!"
    print(" [PASS] Correlation in [-1.0, 1.0] (measures directional alignment along grain)")

    # Energy must be within [0, 1]
    assert 0.0 <= summary["energy_mean"] <= 1.0, f"Invalid Energy: {summary['energy_mean']}"
    assert all(0.0 <= e <= 1.0 for e in np.array(details["energy"]["raw_matrix"]).flatten()), "Energy out of [0, 1]!"
    print(" [PASS] Energy in [0.0, 1.0] (measures texture uniformity & surface smoothness)")

    # Homogeneity must be within [0, 1]
    assert 0.0 <= summary["homogeneity_mean"] <= 1.0, f"Invalid Homogeneity: {summary['homogeneity_mean']}"
    assert all(0.0 <= h <= 1.0 for h in np.array(details["homogeneity"]["raw_matrix"]).flatten()), "Homogeneity out of [0, 1]!"
    print(" [PASS] Homogeneity in [0.0, 1.0] (measures closeness of grain pixel transitions)")

    # Standard deviations must be non-negative
    assert all(summary[f"{f}_std"] >= 0.0 for f in ["contrast", "correlation", "energy", "homogeneity"]), "Negative std dev!"
    print(" [PASS] Standard deviations are valid (>= 0.0)")

    print("\n" + "=" * 70)
    print(" ALL GLCM FEATURE EXTRACTION CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 70)


if __name__ == "__main__":
    test_glcm_extraction()
