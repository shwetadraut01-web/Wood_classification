"""
Wood Species Prediction Tool (Single Image CLI)
================================================
Loads the trained SVM pipeline and performs wood species prediction on a single image.

Usage:
    python3 scripts/predict_species.py <path_to_wood_image>

Example:
    python3 scripts/predict_species.py data/wood_species/splits/test/oak/test_45_4.jpg

Pipeline:
    Image -> Preprocessing (Resize 512x512, Grayscale, Bilateral Filter)
          -> Handcrafted GLCM + LBP Extraction (24-dim vector)
          -> StandardScaler (fitted during training)
          -> RBF Support Vector Classifier
          -> Predicted Species & Raw Decision Scores
"""

import argparse
import os
import sys
import numpy as np

# Ensure project root in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import joblib

from src.preprocessing import preprocess_image
from src.texture_features import extract_texture_vector, FEATURE_NAMES

# Default Model Location
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "species_svm_pipeline.joblib")


def predict_wood_species(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Loads an image, extracts 24 texture features, and runs the trained SVM classifier.

    Parameters:
        image_path (str): Path to input image file.
        model_path (str): Path to trained SVM pipeline .joblib file.

    Returns:
        dict: Prediction results including predicted species and decision scores.
    """
    # 1. Validate Image Path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    if not os.path.isfile(image_path):
        raise ValueError(f"Provided path is not a regular file: '{image_path}'")

    # 2. Validate Model Path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model pipeline not found at '{model_path}'.\n"
            "Please run 'python3 scripts/train_species_svm.py' to train and save the model."
        )

    # 3. Load & Preprocess Image
    try:
        preprocessed_gray = preprocess_image(image_path, target_size=(512, 512), denoise=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load or preprocess image '{image_path}': {e}")

    # 4. Extract Handcrafted 24-Dimensional Texture Features (GLCM + LBP)
    try:
        feature_vector = extract_texture_vector(preprocessed_gray)
        if len(feature_vector) != 24 or np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
            raise ValueError("Extracted feature vector is invalid or contains non-finite values.")
        X = feature_vector.reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for '{image_path}': {e}")

    # 5. Load Trained Pipeline (StandardScaler + SVC)
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load SVM model pipeline from '{model_path}': {e}")

    # 6. Predict Species
    try:
        predicted_class = pipeline.predict(X)[0]
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # 7. Compute Decision Scores (Hyperplane Margins)
    decision_scores = {}
    if hasattr(pipeline, "decision_function"):
        try:
            scores = pipeline.decision_function(X)[0]
            classes = pipeline.classes_
            for cls_name, score in zip(classes, scores):
                decision_scores[str(cls_name)] = float(score)
        except Exception:
            decision_scores = {}

    return {
        "image_path": image_path,
        "filename": os.path.basename(image_path),
        "predicted_species": predicted_class,
        "decision_scores": decision_scores,
        "feature_count": len(feature_vector),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict wood species from a single image using the trained baseline SVM classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 scripts/predict_species.py data/wood_species/splits/test/oak/test_45_4.jpg
        """,
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the wood surface image (JPEG, PNG, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Optional path to the trained model .joblib file (default: models/species_svm_pipeline.joblib)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    try:
        result = predict_wood_species(args.image_path, model_path=args.model)
    except Exception as err:
        print("\n" + "=" * 55, file=sys.stderr)
        print(" ERROR: Prediction Failed", file=sys.stderr)
        print("=" * 55, file=sys.stderr)
        print(f" {err}\n", file=sys.stderr)
        sys.exit(1)

    # Display Clean Output
    print("\n" + "=" * 55)
    print(" Wood Species Prediction ")
    print("=" * 55)
    print(f" Image             : {result['filename']}")
    print(f" Full Path         : {result['image_path']}")
    print(f" Predicted Species : {result['predicted_species'].capitalize()}")
    print("-" * 55)

    if result["decision_scores"]:
        print(" Raw SVM Decision Scores (Hyperplane Margins):")
        print(" [Note: Higher positive score indicates stronger margin separation;")
        print("  these are geometric distances, NOT probabilities or confidence %]\n")
        # Sort classes by decision score descending
        sorted_scores = sorted(
            result["decision_scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for cls_name, score in sorted_scores:
            indicator = " <-- Predicted" if cls_name == result["predicted_species"] else ""
            sign = "+" if score >= 0 else ""
            print(f"   - {cls_name.capitalize():<12}: {sign}{score:8.4f}{indicator}")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
