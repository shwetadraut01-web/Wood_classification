# Wood Analysis and Furniture Recommendation System

A modular system for wood surface inspection, texture analysis, and recommendation.

---

## Current Scope: Module 1 — Wood Texture Analysis

Module 1 focuses strictly on **Wood Texture Feature Extraction & Scoring** using classical computer vision methods:
- **GLCM (Gray-Level Co-occurrence Matrix)** for spatial gray-level relationships (Contrast, Correlation, Energy, Homogeneity).
- **LBP (Local Binary Patterns)** for micro-texture representation and pattern distribution histograms.
- **Texture Scoring** to evaluate grain regularity, surface uniformity, and roughness.

> **Note:** This module relies entirely on mathematical feature extraction techniques. It does not use deep learning or machine learning training models.

---

## Project Structure

```text
wood-analysis-system/
├── requirements.txt         # Core dependencies (OpenCV, NumPy, scikit-image, Matplotlib)
├── README.md                # Project documentation and Module 1 overview
├── main.py                  # Entry point for running wood texture analysis
├── src/                     # Source code package for Module 1 pipeline components
│   └── __init__.py          # Package initialization file
├── tests/                   # Directory for test scripts and verification routines
└── data/                    # Data directory
    └── sample_images/       # Storage for sample wood texture images
```

---

## Setup & Installation

### 1. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
# .\venv\Scripts\activate      # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
