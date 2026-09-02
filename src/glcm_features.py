"""
Module 1: Wood Texture Analysis - GLCM Feature Extraction
=========================================================
This module extracts second-order statistical texture features from wood
surfaces using the Gray-Level Co-occurrence Matrix (GLCM).

GLCM evaluates the spatial distribution and frequency of pixel gray-level pairs
at specified spatial distances (d) and orientations (theta).

Key Features Extracted:
1. Contrast    : Local intensity variation / sharpness of wood grain lines.
2. Correlation : Linear dependency of gray levels along grain direction.
3. Energy      : Uniformity and smoothness of the wood surface (ASM square root).
4. Homogeneity : Closeness of the distribution of elements to the GLCM diagonal.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# Standard default orientations (in radians and degrees)
DEFAULT_DISTANCES = [1, 3, 5]
DEFAULT_ANGLES_RAD = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DEFAULT_ANGLES_DEG = [0, 45, 90, 135]


def quantize_gray_image(gray_image: np.ndarray, num_levels: int = 64) -> np.ndarray:
    """
    Quantizes an 8-bit grayscale image (256 gray levels) into a reduced number of bins.

    Why Quantization is Useful in Texture Analysis:
    -----------------------------------------------
    - Reduces the GLCM matrix size from (256 x 256) to (N x N), which speeds up
      matrix computation and prevents sparse, noisy co-occurrence matrices.
    - For wood analysis, 64 or 256 levels capture rich texture details without loss of grain patterns.

    Parameters:
        gray_image (np.ndarray): 2D 8-bit grayscale image with values in [0, 255].
        num_levels (int): Desired number of gray levels (e.g. 16, 32, 64, or 256).

    Returns:
        np.ndarray: Quantized image with integer values in [0, num_levels - 1].
    """
    if num_levels == 256:
        return gray_image.astype(np.uint8)

    if not (2 <= num_levels <= 256):
        raise ValueError(f"num_levels must be between 2 and 256, got {num_levels}.")

    # Linear binning from [0, 255] to [0, num_levels - 1]
    step = 256.0 / num_levels
    quantized = np.floor(gray_image / step).astype(np.uint8)
    return np.clip(quantized, 0, num_levels - 1)


def compute_glcm(
    gray_image: np.ndarray,
    distances: List[int] = DEFAULT_DISTANCES,
    angles: List[float] = DEFAULT_ANGLES_RAD,
    levels: int = 256,
    symmetric: bool = True,
    normed: bool = True,
) -> np.ndarray:
    """
    Computes the normalized Gray-Level Co-occurrence Matrix (GLCM) for an image.

    Parameters:
        gray_image (np.ndarray): 2D single-channel grayscale image (H, W).
        distances (list of int): Pixel separation distances (e.g. [1, 3, 5]).
        angles (list of float): Displacement angles in radians (e.g. [0, pi/4, pi/2, 3pi/4]).
        levels (int): Number of gray levels (e.g. 64 or 256).
        symmetric (bool): If True, pairs (i, j) and (j, i) are both counted. Default True.
        normed (bool): If True, normalizes GLCM so sum of elements equals 1. Default True.

    Returns:
        np.ndarray: 4D array of shape (levels, levels, len(distances), len(angles)).
    """
    if not isinstance(gray_image, np.ndarray) or len(gray_image.shape) != 2:
        raise ValueError("Input gray_image must be a 2D single-channel numpy array.")

    # Quantize image if levels < 256
    if levels < 256:
        input_image = quantize_gray_image(gray_image, num_levels=levels)
    else:
        input_image = gray_image.astype(np.uint8)

    # Compute GLCM matrix using scikit-image
    glcm_matrix = graycomatrix(
        input_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )
    return glcm_matrix


def extract_glcm_features(
    gray_image: np.ndarray,
    distances: List[int] = DEFAULT_DISTANCES,
    angles: List[float] = DEFAULT_ANGLES_RAD,
    angles_deg: Optional[List[int]] = None,
    levels: int = 256,
) -> Dict:
    """
    Extracts comprehensive GLCM features (Contrast, Correlation, Energy, Homogeneity)
    with per-distance, per-angle, and aggregated statistics (mean & standard deviation).

    Parameters:
        gray_image (np.ndarray): 2D single-channel grayscale image (H, W).
        distances (list of int): Pixel distances to analyze (default: [1, 3, 5]).
        angles (list of float): Angles in radians (default: [0, pi/4, pi/2, 3pi/4]).
        angles_deg (list of int, optional): Angle labels in degrees for reporting.
        levels (int): Gray levels to use in GLCM (default: 256).

    Returns:
        dict: Structured dictionary containing:
            - 'summary': Aggregated mean and standard deviation for each feature.
            - 'details': Detailed 2D matrices, per-distance means, and per-angle means.
            - 'config': Configuration parameters used for extraction.
    """
    if angles_deg is None:
        angles_deg = [int(round(np.degrees(a))) for a in angles]

    # Step 1: Compute GLCM Matrix
    glcm = compute_glcm(
        gray_image=gray_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    # Step 2: Extract the 4 Core GLCM Properties
    # Each property returns a 2D array of shape (num_distances, num_angles)
    contrast_matrix = graycoprops(glcm, "contrast")
    correlation_matrix = graycoprops(glcm, "correlation")
    energy_matrix = graycoprops(glcm, "energy")
    homogeneity_matrix = graycoprops(glcm, "homogeneity")

    # Optional complementary properties
    dissimilarity_matrix = graycoprops(glcm, "dissimilarity")
    asm_matrix = graycoprops(glcm, "ASM")

    # Step 3: Helper to package matrix into structured details
    def _package_feature(feature_matrix: np.ndarray) -> Dict:
        # Per-distance average (averaged across all angles)
        by_distance = {
            f"distance_{d}px": float(np.mean(feature_matrix[d_idx, :]))
            for d_idx, d in enumerate(distances)
        }
        # Per-angle average (averaged across all distances)
        by_angle = {
            f"angle_{deg}deg": float(np.mean(feature_matrix[:, a_idx]))
            for a_idx, deg in enumerate(angles_deg)
        }
        return {
            "mean": float(np.mean(feature_matrix)),
            "std": float(np.std(feature_matrix)),
            "min": float(np.min(feature_matrix)),
            "max": float(np.max(feature_matrix)),
            "raw_matrix": feature_matrix.tolist(),
            "by_distance": by_distance,
            "by_angle": by_angle,
        }

    contrast_data = _package_feature(contrast_matrix)
    correlation_data = _package_feature(correlation_matrix)
    energy_data = _package_feature(energy_matrix)
    homogeneity_data = _package_feature(homogeneity_matrix)
    dissimilarity_data = _package_feature(dissimilarity_matrix)
    asm_data = _package_feature(asm_matrix)

    # Step 4: Construct structured response
    result = {
        "summary": {
            "contrast_mean": contrast_data["mean"],
            "contrast_std": contrast_data["std"],
            "correlation_mean": correlation_data["mean"],
            "correlation_std": correlation_data["std"],
            "energy_mean": energy_data["mean"],
            "energy_std": energy_data["std"],
            "homogeneity_mean": homogeneity_data["mean"],
            "homogeneity_std": homogeneity_data["std"],
            "dissimilarity_mean": dissimilarity_data["mean"],
            "dissimilarity_std": dissimilarity_data["std"],
            "asm_mean": asm_data["mean"],
            "asm_std": asm_data["std"],
        },
        "details": {
            "contrast": contrast_data,
            "correlation": correlation_data,
            "energy": energy_data,
            "homogeneity": homogeneity_data,
            "dissimilarity": dissimilarity_data,
            "asm": asm_data,
        },
        "config": {
            "distances": distances,
            "angles_deg": angles_deg,
            "angles_rad": [float(a) for a in angles],
            "levels": levels,
        },
    }

    return result
