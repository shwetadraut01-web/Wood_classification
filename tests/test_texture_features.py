"""
Test & Verification Script for Combined Texture Features (Module 1)
===================================================================
Tests the integration of GLCM and LBP feature extractors:
1. Loads sample wood texture image.
2. Runs the existing preprocessing pipeline.
3. Extracts combined 24-dimensional GLCM + LBP texture feature representation.
4. Verifies all required keys exist and all values are finite (no NaN, no Inf).
5. Validates LBP histogram probability bounds.
6. Prints a formatted table of all 24 scalar features.
"""

import os
import sys
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image
from src.texture_features import extract_texture_features, extract_texture_vector, get_feature_names, FEATURE_NAMES


def test_combined_texture_features():
    print("=" * 78)
    print(" Module 1: Wood Texture Analysis - GLCM + LBP Feature Combination Test ")
    print("=" * 78)

    sample_image_path = os.path.join(PROJECT_ROOT, "data", "sample_images", "sample_wood.png")

    if not os.path.exists(sample_image_path):
        from tests.test_preprocessing import generate_sample_wood_image
        print(f"Creating sample wood image at: {sample_image_path}")
        generate_sample_wood_image(sample_image_path)

    # 1. Run Preprocessing
    print(f"\n[Step 1] Loading and preprocessing sample image...")
    preprocessed_gray = preprocess_image(sample_image_path, target_size=(512, 512), denoise=True)
    print(f"  - Preprocessed shape : {preprocessed_gray.shape}")
    print(f"  - Data type          : {preprocessed_gray.dtype}")

    # 2. Extract Combined Texture Features
    print(f"\n[Step 2] Extracting combined GLCM + LBP features...")
    features = extract_texture_features(preprocessed_gray)
    feature_vector = extract_texture_vector(preprocessed_gray)

    # 3. Print Complete Combined Feature Table
    print("\n" + "-" * 78)
    print(" COMBINED TEXTURE FEATURE DICTIONARY (24 Scalar Features)")
    print("-" * 78)
    print(f" {'#':<3} | {'Feature Name':<28} | {'Value':<14} | {'Group'}")
    print("-" * 78)

    for i, name in enumerate(FEATURE_NAMES):
        val = features[name]
        if name.startswith("glcm"):
            grp = "GLCM (Spatial Correlation)"
        elif "bin" in name:
            grp = "LBP (Micro-Pattern Bins)"
        else:
            grp = "LBP (Statistical Moments)"
        print(f" {i+1:<3} | {name:<28} | {val:<14.6f} | {grp}")
    print("-" * 78)
    print(f" Total Combined Scalar Features: {len(features)}")
    print("-" * 78)

    # 4. Assertions & Integrity Validations
    print("\n" + "=" * 78)
    print(" VERIFYING DATA INTEGRITY & NUMERICAL BOUNDS...")
    print("=" * 78)

    # Verify total feature count
    expected_count = 24
    assert len(features) == expected_count, f"Expected {expected_count} features, got {len(features)}"
    assert len(feature_vector) == expected_count, f"Feature vector length mismatch: {len(feature_vector)}"
    print(f" [PASS] Total scalar feature count == {expected_count}")

    # Verify all expected keys exist
    for expected_key in FEATURE_NAMES:
        assert expected_key in features, f"Missing feature key: '{expected_key}'"
    print(" [PASS] All 24 required feature keys are present in dictionary")

    # Verify all features are finite (no NaN, no Inf)
    for k, v in features.items():
        assert isinstance(v, (int, float)), f"Feature '{k}' is not numeric! Type: {type(v)}"
        assert not np.isnan(v), f"Feature '{k}' is NaN!"
        assert not np.isinf(v), f"Feature '{k}' is Infinite!"
    print(" [PASS] All 24 numeric scalar features are finite (zero NaN / zero Infinite)")

    # Verify LBP histogram bins
    lbp_bins = [features[f"lbp_bin_{i}"] for i in range(10)]
    assert len(lbp_bins) == 10, "Expected 10 LBP histogram bins"
    assert all(p >= 0.0 for p in lbp_bins), "Negative LBP probability found!"
    assert all(p <= 1.0 for p in lbp_bins), "LBP probability > 1.0 found!"
    bin_sum = sum(lbp_bins)
    assert np.isclose(bin_sum, 1.0, atol=1e-5), f"LBP histogram sum != 1.0! Got {bin_sum}"
    print(f" [PASS] LBP histogram bins are valid probabilities in [0, 1] (Sum = {bin_sum:.6f})")

    # Verify GLCM properties within bounds
    assert features["glcm_contrast_mean"] >= 0.0, "GLCM contrast < 0"
    assert -1.0 <= features["glcm_correlation_mean"] <= 1.0, "GLCM correlation out of [-1, 1]"
    assert 0.0 <= features["glcm_energy_mean"] <= 1.0, "GLCM energy out of [0, 1]"
    assert 0.0 <= features["glcm_homogeneity_mean"] <= 1.0, "GLCM homogeneity out of [0, 1]"
    print(" [PASS] GLCM properties conform to theoretical mathematical bounds")

    print("\n" + "=" * 78)
    print(" ALL COMBINED TEXTURE FEATURE CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 78)


if __name__ == "__main__":
    test_combined_texture_features()
