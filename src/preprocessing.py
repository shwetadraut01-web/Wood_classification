"""
Module 1: Wood Texture Analysis - Image Preprocessing
=====================================================
This module provides functions for loading, validating, resizing,
and converting wood images to preprocessed grayscale format for
subsequent GLCM and LBP texture feature extraction.

Pipeline:
    Input Image -> Load & Validate -> Resize -> Grayscale Conversion -> Noise Reduction -> Preprocessed Image
"""

import os
from typing import Tuple, Union
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path and validates its integrity.

    Parameters:
        image_path (str): File path to the input wood image.

    Returns:
        np.ndarray: Loaded image in OpenCV BGR format (or grayscale if 1-channel).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If OpenCV fails to decode or read the image file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found at: '{image_path}'. "
            f"Please verify the path and file name."
        )

    # Load image using OpenCV (loads as BGR by default)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Failed to load image from: '{image_path}'. "
            f"The file may be corrupted or in an unsupported image format."
        )

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a color image (BGR or BGRA) into an 8-bit single-channel grayscale image.

    Parameters:
        image (np.ndarray): Input image array (2D grayscale or 3/4-channel color).

    Returns:
        np.ndarray: 8-bit single-channel grayscale image of shape (H, W).

    Raises:
        TypeError: If input is not a valid numpy ndarray.
        ValueError: If input array has an unsupported shape.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a numpy ndarray, got {type(image)}.")

    # If already a 2D single-channel image, return as uint8
    if len(image.shape) == 2:
        return image.astype(np.uint8)

    # If 3D single-channel (H, W, 1), squeeze to (H, W)
    if len(image.shape) == 3 and image.shape[2] == 1:
        return image.squeeze(axis=2).astype(np.uint8)

    # If 3-channel (BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If 4-channel (BGRA)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(
        f"Unsupported image shape: {image.shape}. "
        f"Expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
    )


def preprocess_image(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
    denoise: bool = True
) -> np.ndarray:
    """
    Standardizes and preprocesses a wood image for texture feature extraction.

    Steps:
    1. If a file path string is passed, load and validate the image.
    2. Resize to a consistent target dimension using edge-preserving interpolation.
    3. Convert to 8-bit single-channel grayscale.
    4. Optionally apply mild edge-preserving noise reduction (Bilateral Filter)
       to suppress sensor noise without blurring the fine wood grain patterns.

    Parameters:
        image (str or np.ndarray): File path to image or an existing image array.
        target_size (tuple of int): Desired (width, height) output dimensions. Default (512, 512).
        denoise (bool): If True, applies mild edge-preserving noise reduction. Default True.

    Returns:
        np.ndarray: Preprocessed 8-bit grayscale image of shape (height, width).
    """
    # Step 1: Load image if path string is provided
    if isinstance(image, str):
        img = load_image(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError(
            f"Invalid input type: {type(image)}. "
            f"Expected a file path string (str) or a numpy array (np.ndarray)."
        )

    # Step 2: Resize to consistent dimensions
    target_w, target_h = target_size
    current_h, current_w = img.shape[:2]

    if (current_w, current_h) != (target_w, target_h):
        # Choose appropriate interpolation: INTER_AREA for downscaling, INTER_CUBIC for upscaling
        if current_w > target_w or current_h > target_h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Step 3: Convert to Grayscale
    gray = to_grayscale(img)

    # Step 4: Mild Noise Reduction
    # Bilateral filter is ideal for wood texture analysis because it smooths noise
    # while preserving sharp edges and grain boundaries.
    if denoise:
        # d=5 (neighborhood diameter), sigmaColor=25 (color space smoothness), sigmaSpace=25 (coordinate space)
        processed = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    else:
        processed = gray

    # Ensure output is strictly uint8 in range [0, 255]
    return processed.astype(np.uint8)
