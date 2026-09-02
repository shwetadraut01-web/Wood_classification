"""
Test Suite for Wood Species Baseline SVM Classifier
===================================================
Tests functionality and robustness of the SVM classification pipeline:
1. Feature extractor produces 24-dimensional feature vector.
2. Feature vectors are 100% finite (no NaN, no Inf).
3. Pipeline (StandardScaler + SVC) trains on a subset without error.
4. Predictions contain only valid project class labels.
5. Predictions have the correct shape and sample count.
6. Pipeline serializes and deserializes cleanly with joblib.
"""

import os
import sys
import tempfile
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure project root in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image
from src.texture_features import extract_texture_vector, FEATURE_NAMES

PROJECT_CLASSES = ["oak", "pine", "walnut", "sheesham", "mahogany"]


def test_svm_pipeline_functionality():
    print("=" * 75)
    print(" Unit Test: Wood Species Baseline SVM Classifier ")
    print("=" * 75)

    # 1. Test Feature Extraction on Sample Image
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample_images", "sample_wood.png")
    if not os.path.exists(sample_path):
        from tests.test_preprocessing import generate_sample_wood_image
        generate_sample_wood_image(sample_path)

    gray = preprocess_image(sample_path, target_size=(512, 512), denoise=True)
    feat_vec = extract_texture_vector(gray)

    print(f"\n[Test 1] Feature vector dimension verification...")
    assert isinstance(feat_vec, np.ndarray), "Feature vector is not a numpy array"
    assert feat_vec.shape == (24,), f"Expected shape (24,), got {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "Feature vector contains NaN!"
    assert not np.isinf(feat_vec).any(), "Feature vector contains Inf!"
    print(f" [PASS] Extracted feature vector has shape {feat_vec.shape} and is 100% finite.")

    # 2. Test Pipeline Training on a Small Toy Multi-Class Dataset
    print(f"\n[Test 2] Testing Pipeline training on toy multi-class dataset...")
    np.random.seed(42)
    n_samples_per_class = 10
    X_dummy = []
    y_dummy = []

    for idx, cls in enumerate(PROJECT_CLASSES):
        # Create distinct cluster per class with 24 features
        class_features = np.random.normal(loc=idx * 2.5, scale=1.0, size=(n_samples_per_class, 24))
        X_dummy.append(class_features)
        y_dummy.extend([cls] * n_samples_per_class)

    X_train = np.vstack(X_dummy)
    y_train = np.array(y_dummy, dtype=object)

    # Construct standard Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)),
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)
    print(" [PASS] Pipeline (StandardScaler + SVC) fitted successfully without error.")

    # 3. Test Prediction Validity
    print(f"\n[Test 3] Testing prediction outputs...")
    X_test_dummy = np.random.normal(loc=2.0, scale=1.0, size=(15, 24))
    y_pred = pipeline.predict(X_test_dummy)

    assert len(y_pred) == 15, f"Expected 15 predictions, got {len(y_pred)}"
    assert all(pred in PROJECT_CLASSES for pred in y_pred), f"Unexpected class label in {y_pred}"
    print(f" [PASS] Predictions have correct shape ({len(y_pred)}) and contain only valid labels: {set(y_pred)}")

    # 4. Test Model Serialization (Joblib)
    print(f"\n[Test 4] Testing model serialization and persistence...")
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_model_path = tmp_file.name

    try:
        joblib.dump(pipeline, tmp_model_path)
        loaded_pipeline = joblib.load(tmp_model_path)
        y_pred_loaded = loaded_pipeline.predict(X_test_dummy)
        np.testing.assert_array_equal(y_pred, y_pred_loaded)
        print(" [PASS] Pipeline serialized and reloaded with identical predictions.")
    finally:
        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

    print("\n" + "=" * 75)
    print(" ALL SVM CLASSIFIER UNIT TESTS PASSED SUCCESSFULLY! ")
    print("=" * 75)


if __name__ == "__main__":
    test_svm_pipeline_functionality()
