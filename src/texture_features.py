"""
Module 1: Wood Texture Analysis - Combined GLCM + LBP Feature Representation
=============================================================================
This module combines second-order statistical texture features (GLCM) and
micro-structural local binary pattern features (LBP) into a unified,
flat feature representation for wood surface analysis.

Combined Feature Vector (24 Scalar Features):
--------------------------------------------
1. GLCM Features (8 features):
   - glcm_contrast_mean       : Local intensity variation / grain groove depth
   - glcm_contrast_std        : Angular/distance variance of contrast
   - glcm_correlation_mean    : Linear dependency along grain lines
   - glcm_correlation_std     : Directional variance of grain correlation
   - glcm_energy_mean         : Texture uniformity / surface smoothness
   - glcm_energy_std          : Directional variance of energy
   - glcm_homogeneity_mean    : Closeness of gray levels across pairs
   - glcm_homogeneity_std     : Directional variance of homogeneity

2. LBP Statistical Moments (6 features):
   - lbp_histogram_mean       : Mean frequency of uniform pattern bins
   - lbp_histogram_std        : Standard deviation of histogram distribution
   - lbp_histogram_entropy    : Shannon entropy in bits (micro-texture complexity)
   - lbp_histogram_energy     : Sum of squared probabilities (uniformity)
   - lbp_uniform_ratio        : Fraction of structured/uniform micro-patterns
   - lbp_non_uniform_ratio    : Fraction of high-frequency noise/random patterns

3. LBP Normalized Histogram Bins (10 features):
   - lbp_bin_0 through lbp_bin_9 : Individual pattern probabilities summing to 1.0
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.glcm_features import extract_glcm_features
from src.lbp_features import extract_lbp_features


FEATURE_NAMES: List[str] = [
    # GLCM features (8)
    "glcm_contrast_mean",
    "glcm_contrast_std",
    "glcm_correlation_mean",
    "glcm_correlation_std",
    "glcm_energy_mean",
    "glcm_energy_std",
    "glcm_homogeneity_mean",
    "glcm_homogeneity_std",
    # LBP statistical moments (6)
    "lbp_histogram_mean",
    "lbp_histogram_std",
    "lbp_histogram_entropy",
    "lbp_histogram_energy",
    "lbp_uniform_ratio",
    "lbp_non_uniform_ratio",
    # LBP histogram bins (10)
    "lbp_bin_0",
    "lbp_bin_1",
    "lbp_bin_2",
    "lbp_bin_3",
    "lbp_bin_4",
    "lbp_bin_5",
    "lbp_bin_6",
    "lbp_bin_7",
    "lbp_bin_8",
    "lbp_bin_9",
]


def get_feature_names() -> List[str]:
    """
    Returns the ordered list of all 24 combined texture feature names.
    """
    return list(FEATURE_NAMES)


def extract_texture_features(
    gray_image: np.ndarray,
    glcm_distances: Optional[List[int]] = None,
    glcm_levels: int = 256,
    lbp_radius: int = 1,
    lbp_points: int = 8,
) -> Dict[str, float]:
    """
    Extracts and combines GLCM and LBP texture features into a single flat dictionary.

    Parameters:
        gray_image (np.ndarray): 2D preprocessed 8-bit grayscale image (H, W).
        glcm_distances (list of int, optional): Pixel distances for GLCM (default: [1, 3, 5]).
        glcm_levels (int): Number of gray levels for GLCM (default: 256).
        lbp_radius (int): Circular neighborhood radius for LBP (default: 1).
        lbp_points (int): Number of circular sampling points for LBP (default: 8).

    Returns:
        dict: Flat dictionary containing all 24 scalar numeric features.
    """
    if not isinstance(gray_image, np.ndarray) or len(gray_image.shape) != 2:
        raise ValueError(
            f"Expected 2D single-channel numpy array, got shape {getattr(gray_image, 'shape', type(gray_image))}."
        )

    # 1. Extract GLCM features
    if glcm_distances is None:
        glcm_distances = [1, 3, 5]

    glcm_raw = extract_glcm_features(
        gray_image,
        distances=glcm_distances,
        levels=glcm_levels,
    )
    glcm_summary = glcm_raw["summary"]

    # 2. Extract LBP features
    lbp_raw = extract_lbp_features(
        gray_image,
        radius=lbp_radius,
        points=lbp_points,
        method="uniform",
    )

    # 3. Assemble unified flat feature dictionary
    features: Dict[str, float] = {
        # GLCM Core Properties (8)
        "glcm_contrast_mean": float(glcm_summary["contrast_mean"]),
        "glcm_contrast_std": float(glcm_summary["contrast_std"]),
        "glcm_correlation_mean": float(glcm_summary["correlation_mean"]),
        "glcm_correlation_std": float(glcm_summary["correlation_std"]),
        "glcm_energy_mean": float(glcm_summary["energy_mean"]),
        "glcm_energy_std": float(glcm_summary["energy_std"]),
        "glcm_homogeneity_mean": float(glcm_summary["homogeneity_mean"]),
        "glcm_homogeneity_std": float(glcm_summary["homogeneity_std"]),
        # LBP Statistical Moments (6)
        "lbp_histogram_mean": float(lbp_raw["histogram_mean"]),
        "lbp_histogram_std": float(lbp_raw["histogram_std"]),
        "lbp_histogram_entropy": float(lbp_raw["histogram_entropy"]),
        "lbp_histogram_energy": float(lbp_raw["histogram_energy"]),
        "lbp_uniform_ratio": float(lbp_raw["uniform_ratio"]),
        "lbp_non_uniform_ratio": float(lbp_raw["non_uniform_ratio"]),
    }

    # LBP Histogram Bins (10)
    for bin_idx, prob in enumerate(lbp_raw["histogram"]):
        features[f"lbp_bin_{bin_idx}"] = float(prob)

    return features


def extract_texture_vector(
    gray_image: np.ndarray,
    glcm_distances: Optional[List[int]] = None,
    glcm_levels: int = 256,
    lbp_radius: int = 1,
    lbp_points: int = 8,
) -> np.ndarray:
    """
    Extracts texture features as a 1D NumPy array in the standardized feature order.

    Returns:
        np.ndarray: 1D float64 array of shape (24,).
    """
    feat_dict = extract_texture_features(
        gray_image,
        glcm_distances=glcm_distances,
        glcm_levels=glcm_levels,
        lbp_radius=lbp_radius,
        lbp_points=lbp_points,
    )
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
