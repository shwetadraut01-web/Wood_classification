"""
Demo & Test Script for Preprocessing (Module 1)
==============================================
Demonstrates and validates:
1. Loading an image from data/sample_images/
2. Converting and preprocessing the wood texture image
3. Displaying and saving the preprocessed output image
4. Verifying image dimensions, channel count, and data types
"""

import os
import sys
import cv2
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, to_grayscale, preprocess_image


def generate_sample_wood_image(output_path: str, width: int = 600, height: int = 600):
    """
    Generates a realistic synthetic wood grain texture for testing when no real image is provided.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create coordinate grid
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Simulate wavy wood grain rings
    rings = np.sin(x * 0.05 + 2.5 * np.sin(y * 0.02) + np.sin(x * 0.01 + y * 0.01))
    
    # Add high-frequency micro-texture / grain fiber noise
    np.random.seed(42)
    fibers = np.random.normal(0, 0.08, size=(height, width)).astype(np.float32)
    
    grain = (rings + 1.0) / 2.0  # Normalize to [0, 1]
    grain = np.clip(grain + fibers, 0, 1)
    
    # Apply warm wood tone (BGR format for OpenCV)
    # Dark brown grain: B=35, G=65, R=110 | Light brown base: B=70, G=130, R=190
    base_b, base_g, base_r = 50, 100, 160
    dark_b, dark_g, dark_r = 25, 55, 95
    
    wood_b = (dark_b + grain * (base_b - dark_b)).astype(np.uint8)
    wood_g = (dark_g + grain * (base_g - dark_g)).astype(np.uint8)
    wood_r = (dark_r + grain * (base_r - dark_r)).astype(np.uint8)
    
    wood_img = cv2.merge([wood_b, wood_g, wood_r])
    cv2.imwrite(output_path, wood_img)
    return wood_img


def run_preprocessing_demo():
    print("=" * 65)
    print(" Module 1: Wood Texture Analysis - Preprocessing Test & Demo ")
    print("=" * 65)

    sample_dir = os.path.join(PROJECT_ROOT, "data", "sample_images")
    sample_image_path = os.path.join(sample_dir, "sample_wood.png")
    output_processed_path = os.path.join(sample_dir, "sample_wood_preprocessed.png")

    # If sample image doesn't exist, create one
    if not os.path.exists(sample_image_path):
        print(f"Creating sample wood texture image at: {sample_image_path}")
        generate_sample_wood_image(sample_image_path, width=640, height=480)
        print("Sample image successfully created.")

    # 1. Load image
    print(f"\n[Step 1] Loading image from: {sample_image_path}")
    original_img = load_image(sample_image_path)
    print(f"  - Original Dimensions (H, W, C): {original_img.shape}")
    print(f"  - Data Type                    : {original_img.dtype}")
    print(f"  - Pixel Value Range            : [{original_img.min()}, {original_img.max()}]")

    # 2. Test to_grayscale
    print(f"\n[Step 2] Testing grayscale conversion...")
    gray_img = to_grayscale(original_img)
    print(f"  - Grayscale Dimensions (H, W) : {gray_img.shape}")
    print(f"  - Number of Channels           : {1 if len(gray_img.shape) == 2 else gray_img.shape[2]}")
    print(f"  - Data Type                    : {gray_img.dtype}")

    # 3. Test preprocess_image pipeline
    print(f"\n[Step 3] Running full preprocessing pipeline...")
    target_size = (512, 512)
    processed_img = preprocess_image(sample_image_path, target_size=target_size, denoise=True)
    print(f"  - Target Size (W, H)           : {target_size}")
    print(f"  - Processed Dimensions (H, W)  : {processed_img.shape}")
    print(f"  - Data Type                    : {processed_img.dtype}")
    print(f"  - Pixel Value Range            : [{processed_img.min()}, {processed_img.max()}]")

    # 4. Save output
    cv2.imwrite(output_processed_path, processed_img)
    print(f"\n[Step 4] Processed grayscale image saved to:")
    print(f"  -> {output_processed_path}")

    # 5. Test error handling
    print(f"\n[Step 5] Testing error handling for missing file...")
    try:
        load_image("data/sample_images/non_existent_file.png")
    except FileNotFoundError as e:
        print(f"  - Caught expected error: {e}")

    # 6. Test direct in-memory array preprocessing
    print(f"\n[Step 6] Testing direct NumPy array preprocessing...")
    dummy_color = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    processed_array = preprocess_image(dummy_color, target_size=(256, 256))
    print(f"  - Input shape: (300, 400, 3) -> Output shape: {processed_array.shape}, dtype: {processed_array.dtype}")

    # 7. Validation assertions
    assert processed_img.shape == (512, 512), f"Expected shape (512, 512), got {processed_img.shape}"
    assert processed_img.dtype == np.uint8, f"Expected dtype uint8, got {processed_img.dtype}"
    assert len(processed_img.shape) == 2, "Expected 2D single-channel image"
    assert processed_array.shape == (256, 256), f"Expected shape (256, 256), got {processed_array.shape}"

    print("\n" + "=" * 65)
    print(" ALL PREPROCESSING CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 65)


if __name__ == "__main__":
    run_preprocessing_demo()
