"""
Module 1: Wood Texture Analysis - Local Binary Patterns (LBP) Feature Extraction
================================================================================
This module extracts micro-texture, roughness, and pattern distribution features
from wood surfaces using Local Binary Patterns (LBP).

LBP evaluates local spatial neighborhoods around each pixel:
1. Compares each neighbor against the central pixel.
2. Generates a binary code (1 if neighbor >= center, else 0).
3. Using the 'uniform' method, patterns with at most 2 circular 0-1/1-0 transitions
   are mapped to unique bins (0 to P), while all complex/non-uniform patterns
   are grouped into a single final bin (P + 1).

Extracted Features:
- Normalized LBP Histogram (probabilities summing to 1.0)
- Histogram Mean and Standard Deviation
- Shannon Entropy (measures texture complexity/randomness)
- Uniform Pattern Ratio vs. Non-Uniform Noise Ratio
- LBP Energy (Uniformity of pattern distribution)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from skimage.feature import local_binary_pattern


def compute_lbp_map(
    gray_image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
    method: str = "uniform",
) -> np.ndarray:
    """
    Computes the 2D Local Binary Pattern (LBP) texture representation map.

    Parameters:
        gray_image (np.ndarray): 2D single-channel grayscale image (H, W).
        radius (int): Radius of the circular neighborhood (default: 1).
        n_points (int): Number of circularly symmetric neighbor points (default: 8).
        method (str): LBP encoding method ('uniform', 'default', 'ror', 'nri_uniform').
                      Default: 'uniform'.

    Returns:
        np.ndarray: 2D array of LBP code values matching image shape (H, W).
    """
    if not isinstance(gray_image, np.ndarray) or len(gray_image.shape) != 2:
        raise ValueError("Input gray_image must be a 2D single-channel numpy array.")

    # Ensure image is in uint8 format
    img_uint8 = gray_image.astype(np.uint8)

    # Compute LBP map using scikit-image
    lbp_map = local_binary_pattern(
        img_uint8,
        P=n_points,
        R=radius,
        method=method,
    )
    return lbp_map


def compute_lbp_histogram(
    lbp_map: np.ndarray,
    n_points: int = 8,
    method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a normalized probability histogram from an LBP texture map.

    Parameters:
        lbp_map (np.ndarray): 2D LBP texture map.
        n_points (int): Number of neighbor points used when generating the LBP map.
        method (str): LBP method used ('uniform' uses n_points + 2 bins).

    Returns:
        tuple: (normalized_histogram, bin_edges)
               - normalized_histogram: 1D array of probabilities summing to 1.0.
               - bin_edges: Array of bin boundaries.
    """
    if method == "uniform":
        # For uniform LBP, valid codes are 0, 1, ..., P (uniform) and P+1 (non-uniform)
        n_bins = n_points + 2
    else:
        n_bins = 2 ** n_points

    # Compute frequency histogram
    hist, bin_edges = np.histogram(
        lbp_map.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=False,
    )

    # Normalize histogram to probability distribution (sums to 1.0)
    total_pixels = hist.sum()
    if total_pixels > 0:
        normalized_hist = hist.astype(np.float64) / total_pixels
    else:
        normalized_hist = np.zeros(n_bins, dtype=np.float64)

    return normalized_hist, bin_edges


def compute_shannon_entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """
    Computes Shannon Entropy of a probability distribution.
    H = - sum(p_i * log_base(p_i)) for all p_i > 0.

    Higher entropy indicates richer, more complex, or more random texture.
    Lower entropy indicates smooth, homogeneous, or uniform patterns.

    Parameters:
        probabilities (np.ndarray): 1D array of probabilities.
        base (float): Logarithm base (default 2.0 for information bits).

    Returns:
        float: Shannon entropy value >= 0.0.
    """
    # Filter out zero probabilities to avoid log(0)
    non_zero_p = probabilities[probabilities > 0.0]
    if len(non_zero_p) == 0:
        return 0.0

    if base == 2.0:
        entropy = -np.sum(non_zero_p * np.log2(non_zero_p))
    else:
        entropy = -np.sum(non_zero_p * (np.log(non_zero_p) / np.log(base)))

    return float(max(0.0, entropy))


def extract_lbp_features(
    gray_image: np.ndarray,
    radius: int = 1,
    points: int = 8,
    method: str = "uniform",
) -> Dict:
    """
    Extracts comprehensive LBP features, normalized histogram, and statistical metrics.

    Parameters:
        gray_image (np.ndarray): 2D single-channel grayscale image (H, W).
        radius (int): Neighborhood radius in pixels (default: 1).
        points (int): Number of circularly symmetric sampling points (default: 8).
        method (str): LBP method ('uniform' recommended for rotation/noise stability).

    Returns:
        dict: Structured dictionary containing:
            - 'radius': Neighborhood radius used
            - 'number_of_points': Number of sample points P
            - 'method': Encoding method ('uniform')
            - 'num_bins': Total number of histogram bins
            - 'histogram': List of normalized bin probabilities summing to 1.0
            - 'histogram_mean': Mean value of histogram bins
            - 'histogram_std': Standard deviation of histogram bins
            - 'histogram_entropy': Shannon entropy (texture complexity)
            - 'histogram_energy': Sum of squared probabilities (uniformity)
            - 'uniform_ratio': Fraction of patterns that are uniform (grains/edges/spots)
            - 'non_uniform_ratio': Fraction of chaotic/noisy non-uniform patterns
            - 'lbp_map_stats': Spatial mean, std, min, and max of the raw LBP codes
    """
    # Step 1: Generate 2D LBP Texture Map
    lbp_map = compute_lbp_map(
        gray_image=gray_image,
        radius=radius,
        n_points=points,
        method=method,
    )

    # Step 2: Compute Normalized Histogram
    normalized_hist, bin_edges = compute_lbp_histogram(
        lbp_map=lbp_map,
        n_points=points,
        method=method,
    )

    # Step 3: Statistical Metrics on the Histogram
    hist_mean = float(np.mean(normalized_hist))
    hist_std = float(np.std(normalized_hist))
    hist_entropy = compute_shannon_entropy(normalized_hist, base=2.0)
    hist_energy = float(np.sum(normalized_hist ** 2))

    # For uniform LBP: bins 0 to P are uniform patterns, bin P+1 is non-uniform
    if method == "uniform":
        uniform_ratio = float(np.sum(normalized_hist[:points + 1]))
        non_uniform_ratio = float(normalized_hist[points + 1])
    else:
        uniform_ratio = 1.0
        non_uniform_ratio = 0.0

    # Step 4: Spatial LBP Map Statistics
    lbp_map_stats = {
        "mean": float(np.mean(lbp_map)),
        "std": float(np.std(lbp_map)),
        "min": float(np.min(lbp_map)),
        "max": float(np.max(lbp_map)),
    }

    # Step 5: Package into a clean, structured dictionary
    result = {
        "radius": int(radius),
        "points": int(points),
        "number_of_points": int(points),
        "method": str(method),
        "num_bins": len(normalized_hist),
        "histogram": [float(p) for p in normalized_hist],
        "histogram_mean": hist_mean,
        "histogram_std": hist_std,
        "histogram_entropy": hist_entropy,
        "histogram_energy": hist_energy,
        "uniform_ratio": uniform_ratio,
        "non_uniform_ratio": non_uniform_ratio,
        "lbp_map_stats": lbp_map_stats,
    }

    return result
