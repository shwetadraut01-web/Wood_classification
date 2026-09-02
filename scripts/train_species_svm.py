"""
Wood Species Baseline SVM Classifier Training & Evaluation Pipeline
===================================================================
Trains a classical machine learning baseline using handcrafted 24-dimensional
GLCM + LBP texture features extracted from the 5-class wood dataset.

Model Architecture:
- Pipeline: StandardScaler -> SVC(kernel='rbf', class_weight='balanced', random_state=42)
- Tuning: Manual hyperparameter search over C and gamma using the VALIDATION split only.
- Final Evaluation: Single unbiased evaluation on the TEST split after retraining
  the best configuration on (Train + Validation).

Dataset Splits:
- Train: 1,153 images
- Validation: 247 images
- Test: 248 images
"""

import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure project root in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image
from src.texture_features import extract_texture_vector, get_feature_names, FEATURE_NAMES

# Directory Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPECIES_DIR = os.path.join(DATA_DIR, "wood_species")
SPLITS_DIR = os.path.join(SPECIES_DIR, "splits")
METADATA_DIR = os.path.join(SPECIES_DIR, "metadata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_FILE = os.path.join(METADATA_DIR, "texture_features_cache.npz")

PROJECT_CLASSES = ["oak", "pine", "walnut", "sheesham", "mahogany"]
RANDOM_SEED = 42

# Explicit Configuration Metadata
GLCM_CONFIG = {
    "distances_px": [1, 3, 5],
    "angles_deg": [0, 45, 90, 135],
    "quantization_levels": 256,
    "properties_extracted": ["contrast", "correlation", "energy", "homogeneity"],
    "aggregations": ["mean", "std"],
}

LBP_CONFIG = {
    "radius": 1,
    "points": 8,
    "method": "uniform",
    "num_bins": 10,
    "statistical_moments": ["mean", "std", "entropy", "energy", "uniform_ratio", "non_uniform_ratio"],
}


def extract_split_features(
    split_name: str,
    splits_dir: str = SPLITS_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts 24-dim GLCM+LBP texture feature vectors for all images in a split.

    Returns:
        X (np.ndarray): Shape (N, 24) float64 feature matrix.
        y (np.ndarray): Shape (N,) string class label array.
        filenames (list of str): List of relative image filenames.
    """
    split_path = os.path.join(splits_dir, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    X_list = []
    y_list = []
    filenames = []

    print(f" Extracting texture features from split: '{split_name}' ({split_path})...")
    t0 = time.time()

    for cls in PROJECT_CLASSES:
        cls_dir = os.path.join(split_path, cls)
        if not os.path.exists(cls_dir):
            continue

        img_files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
        ])

        for fn in img_files:
            img_path = os.path.join(cls_dir, fn)
            # Preprocessing -> Grayscale -> Bilateral Filter -> GLCM+LBP Vector (24-dim)
            gray = preprocess_image(img_path, target_size=(512, 512), denoise=True)
            vec = extract_texture_vector(gray)

            X_list.append(vec)
            y_list.append(cls)
            filenames.append(f"{cls}/{fn}")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=object)
    t1 = time.time()

    print(f" -> Completed '{split_name}': {len(X)} images in {t1 - t0:.1f}s | Feature Shape: {X.shape}")
    return X, y, filenames


def validate_feature_cache(cache_data: np.lib.npyio.NpzFile) -> bool:
    """
    Validates cache integrity, feature count, classes, and absence of NaNs.
    """
    required_keys = [
        "X_train", "y_train", "files_train",
        "X_val", "y_val", "files_val",
        "X_test", "y_test", "files_test",
        "class_names", "feature_names",
    ]
    for k in required_keys:
        if k not in cache_data:
            print(f" [CACHE WARNING] Missing key '{k}' in cache.")
            return False

    # Validate feature dimensions
    if cache_data["X_train"].shape[1] != len(FEATURE_NAMES):
        print(f" [CACHE WARNING] Feature dimension mismatch: {cache_data['X_train'].shape[1]} vs {len(FEATURE_NAMES)}")
        return False

    # Validate classes
    cached_classes = list(cache_data["class_names"])
    if cached_classes != PROJECT_CLASSES:
        print(f" [CACHE WARNING] Cached classes {cached_classes} != expected {PROJECT_CLASSES}")
        return False

    # Validate finite values (no NaN or Inf)
    for s in ["X_train", "X_val", "X_test"]:
        mat = cache_data[s]
        if np.isnan(mat).any() or np.isinf(mat).any():
            print(f" [CACHE WARNING] Cached matrix '{s}' contains NaN or Inf.")
            return False

    return True


def get_or_create_feature_cache(
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Loads or extracts and caches the feature matrices for train, validation, and test.
    """
    os.makedirs(METADATA_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE) and not force_recompute:
        try:
            data = np.load(CACHE_FILE, allow_pickle=True)
            if validate_feature_cache(data):
                print(f" [CACHE OK] Verified and loaded texture features from: {CACHE_FILE}")
                return {
                    "X_train": data["X_train"],
                    "y_train": data["y_train"],
                    "files_train": data["files_train"].tolist(),
                    "X_val": data["X_val"],
                    "y_val": data["y_val"],
                    "files_val": data["files_val"].tolist(),
                    "X_test": data["X_test"],
                    "y_test": data["y_test"],
                    "files_test": data["files_test"].tolist(),
                }
            else:
                print(" [CACHE NOTICE] Invalid or outdated feature cache. Recomputing from splits...")
        except Exception as e:
            print(f" [CACHE NOTICE] Failed to read existing cache ({e}). Recomputing...")

    print(" Extracting handcrafted GLCM + LBP texture features across all splits...")
    X_train, y_train, files_train = extract_split_features("train")
    X_val, y_val, files_val = extract_split_features("validation")
    X_test, y_test, files_test = extract_split_features("test")

    np.savez_compressed(
        CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        files_train=np.array(files_train, dtype=object),
        X_val=X_val,
        y_val=y_val,
        files_val=np.array(files_val, dtype=object),
        X_test=X_test,
        y_test=y_test,
        files_test=np.array(files_test, dtype=object),
        class_names=np.array(PROJECT_CLASSES, dtype=object),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )
    print(f" Saved verified feature cache to: {CACHE_FILE}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "files_train": files_train,
        "X_val": X_val,
        "y_val": y_val,
        "files_val": files_val,
        "X_test": X_test,
        "y_test": y_test,
        "files_test": files_test,
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Computes classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=PROJECT_CLASSES)

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
    }


def print_confusion_matrix(cm: np.ndarray, labels: List[str]):
    """
    Prints a formatted confusion matrix table.
    """
    header = f" {'Actual \\ Pred':<14} | " + " | ".join([f"{l[:8]:<8}" for l in labels])
    print(" " + "-" * len(header))
    print(header)
    print(" " + "-" * len(header))
    for idx, label in enumerate(labels):
        row_str = " | ".join([f"{cm[idx, j]:<8d}" for j in range(len(labels))])
        print(f" {label:<14} | {row_str}")
    print(" " + "-" * len(header))


def train_and_evaluate():
    print("=" * 82)
    print(" WOOD SPECIES BASELINE SVM CLASSIFIER (GLCM + LBP) ")
    print("=" * 82)

    # 1. Load Feature Matrices
    cache = get_or_create_feature_cache()
    X_train, y_train = cache["X_train"], cache["y_train"]
    X_val, y_val = cache["X_val"], cache["y_val"]
    X_test, y_test = cache["X_test"], cache["y_test"]

    print("\n--- 1. FEATURE MATRIX SHAPES ---")
    print(f" Training Set (X_train)   : {X_train.shape} (Labels: {Counter(y_train)})")
    print(f" Validation Set (X_val)   : {X_val.shape} (Labels: {Counter(y_val)})")
    print(f" Test Set (X_test)         : {X_test.shape} (Labels: {Counter(y_test)})")
    print(f" Features per Sample      : {X_train.shape[1]} (GLCM: 8, LBP: 16)")

    # Assert no NaN or Inf in feature matrices
    for name, mat in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        assert not np.isnan(mat).any(), f"{name} matrix contains NaN!"
        assert not np.isinf(mat).any(), f"{name} matrix contains Inf!"
    print(" [CHECK] All feature matrices are 100% finite (zero NaN / zero Inf).")

    # 2. Manual Hyperparameter Search on VALIDATION Set Only
    print("\n--- 2. MANUAL HYPERPARAMETER SEARCH (Tuned on Validation Set ONLY) ---")
    param_grid_c = [0.1, 1.0, 10.0, 100.0]
    param_grid_gamma = ["scale", 0.01, 0.1, 1.0]

    best_val_f1 = -1.0
    best_params = None
    best_val_metrics = None
    search_results = []

    print(f" {'C':<6} | {'Gamma':<8} | {'Val Accuracy':<14} | {'Val Macro Prec':<16} | {'Val Macro Rec':<15} | {'Val Macro F1':<14}")
    print(" " + "-" * 82)

    for c in param_grid_c:
        for gamma in param_grid_gamma:
            # Construct Pipeline: Scaler fitted only on train data in each candidate evaluation
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    C=c,
                    gamma=gamma,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_SEED,
                )),
            ])

            # Train on TRAIN split
            pipeline.fit(X_train, y_train)

            # Evaluate on VALIDATION split
            y_val_pred = pipeline.predict(X_val)
            metrics = evaluate_predictions(y_val, y_val_pred)

            search_results.append({
                "C": c,
                "gamma": gamma,
                "metrics": metrics,
            })

            gamma_str = f"{gamma}" if isinstance(gamma, str) else f"{gamma:.2f}"
            print(f" {c:<6.1f} | {gamma_str:<8} | {metrics['accuracy']:<14.4f} | {metrics['macro_precision']:<16.4f} | {metrics['macro_recall']:<15.4f} | {metrics['macro_f1']:<14.4f}")

            if metrics["macro_f1"] > best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                best_params = {"C": c, "gamma": gamma}
                best_val_metrics = metrics

    print(" " + "-" * 82)
    print(f" Best Hyperparameters Selected: C = {best_params['C']}, gamma = {best_params['gamma']}")
    print(f" Best Validation Macro F1     : {best_val_metrics['macro_f1']:.4f} (Accuracy: {best_val_metrics['accuracy']:.4f})")

    # Print Validation Confusion Matrix
    print(f"\n--- 3. VALIDATION CONFUSION MATRIX (Best Model: C={best_params['C']}, gamma={best_params['gamma']}) ---")
    val_cm = np.array(best_val_metrics["confusion_matrix"])
    print_confusion_matrix(val_cm, PROJECT_CLASSES)

    # 3. Retrain Best Pipeline on (TRAIN + VALIDATION)
    print("\n--- 4. RETRAINING BEST MODEL ON (TRAIN + VALIDATION) ---")
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    print(f" Combined Train+Val Samples: {X_train_val.shape[0]} images ({Counter(y_train_val)})")

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=best_params["C"],
            gamma=best_params["gamma"],
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_SEED,
        )),
    ])
    final_pipeline.fit(X_train_val, y_train_val)

    # 4. SINGLE UNBIASED EVALUATION ON TEST SPLIT
    print("\n" + "=" * 82)
    print(" 5. FINAL UNBIASED EVALUATION ON TEST SET (248 images) ")
    print("=" * 82)
    print(" [NOTE] Test set was kept completely untouched until this single evaluation.")

    y_test_pred = final_pipeline.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    print("\n Test Set Aggregate Metrics:")
    print(f"  - Accuracy          : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"  - Macro Precision   : {test_metrics['macro_precision']:.4f}")
    print(f"  - Macro Recall      : {test_metrics['macro_recall']:.4f}")
    print(f"  - Macro F1-Score    : {test_metrics['macro_f1']:.4f}")
    print(f"  - Weighted F1-Score : {test_metrics['weighted_f1']:.4f}")

    print("\n Final Test Confusion Matrix:")
    test_cm = np.array(test_metrics["confusion_matrix"])
    print_confusion_matrix(test_cm, PROJECT_CLASSES)

    print("\n Detailed Per-Class Classification Report (Test Set):")
    report = classification_report(
        y_test,
        y_test_pred,
        labels=PROJECT_CLASSES,
        target_names=[c.capitalize() for c in PROJECT_CLASSES],
        digits=4,
        zero_division=0,
    )
    print(report)

    # 5. Save Model and Metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "species_svm_pipeline.joblib")
    joblib.dump(final_pipeline, model_path)
    print(f"\n Model pipeline successfully saved to:\n  -> {model_path}")

    metadata_info = {
        "model_name": "Wood Species Baseline SVM Classifier",
        "model_type": "Support Vector Machine (SVC)",
        "pipeline_stages": ["StandardScaler", "SVC"],
        "hyperparameter_search_method": "Manual hyperparameter search using the validation set",
        "kernel": "rbf",
        "selected_C": best_params["C"],
        "selected_gamma": best_params["gamma"],
        "class_weight": "balanced",
        "random_seed": RANDOM_SEED,
        "feature_count": 24,
        "feature_names": FEATURE_NAMES,
        "glcm_config": GLCM_CONFIG,
        "lbp_config": LBP_CONFIG,
        "class_names": PROJECT_CLASSES,
        "train_image_count": int(X_train.shape[0]),
        "validation_image_count": int(X_val.shape[0]),
        "test_image_count": int(X_test.shape[0]),
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "scientific_notices": [
            "Oak, Pine, and Walnut originate from WOOD-AUTH, while Sheesham and Mahogany originate from GOIMAI. Therefore species and dataset source are currently confounded. Model performance may partly reflect dataset/capture differences rather than wood-species characteristics. Additional cross-source data is recommended before claiming strong real-world generalization.",
            "Leakage prevention is guaranteed at the image/file level; specimen-level leakage cannot be verified from the available metadata.",
            "This is a classical handcrafted feature baseline to benchmark GLCM+LBP separability; it is not a production-ready system.",
        ],
    }

    metadata_path = os.path.join(MODELS_DIR, "species_svm_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata_info, f_meta, indent=2)
    print(f" Model metadata saved to:\n  -> {metadata_path}")

    # 6. Scientific Notices & Disclaimers
    print("\n" + "=" * 82)
    print(" SCIENTIFIC WARNINGS & METHODOLOGY NOTES")
    print("=" * 82)
    print(" 1. Dataset Confounding Warning:")
    print("    'Oak, Pine, and Walnut originate from WOOD-AUTH, while Sheesham and")
    print("     Mahogany originate from GOIMAI. Therefore species and dataset source")
    print("     are currently confounded. Model performance may partly reflect")
    print("     dataset/capture differences rather than wood-species characteristics.")
    print("     Additional cross-source data is recommended before claiming strong")
    print("     real-world generalization.'")
    print("\n 2. Specimen-Level Leakage Notice:")
    print("    'Leakage prevention is guaranteed at the image/file level;")
    print("     specimen-level leakage cannot be verified from the available metadata.'")
    print("\n 3. Baseline Status:")
    print("    'This is a handcrafted classical ML baseline (GLCM+LBP+SVM) to benchmark")
    print("     texture feature separability. It is not a production-grade system.'")
    print("=" * 82)


if __name__ == "__main__":
    train_and_evaluate()
