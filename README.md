
# Cell-Type Composition Prediction from H&E Images

**Authors:** Aditya Deori, Ipsita Pandey, Saurav Ray 
**Institution:** Indian Institute of Technology Ropar 
**Course:** BM616 Machine Learning Project  
**Faculty Mentor:** Dr. Sukrit Gupta 

## Project Overview
Predicting spatially resolved cell-type composition from Hematoxylin-and-Eosin (H&E) stained tissue images is a challenging computational pathology problem. The visual appearance of cells is often ambiguous, and target cell abundances are not independent. 

This project explores two complementary approaches to tackle this:
1.  **Multi-Scale CNNs:** To capture both local and contextual visual information from tissue patches.
2.  **Structured Inter-Cell Refinement:** A two-stage framework where initial predictions are corrected using learned dependencies (co-occurrences and mutual exclusions) between cell types.

## Repository Structure

*   **`configs/`**: Contains `config.json` for model and training parameters.
*   **`data_preprocessing/`**:
    *   **`Image_preprocessing/`**: Notebooks for generating masked images and performing stain normalization.
    *   **`Spot_preprocessing/`**: Notebooks for alignment, identifying invalid spots, and ranking.
    *   **`Final_preprocessing/`**: Notebooks for image tiling and calculating distances.
*   **`scripts/`**: Modular Python scripts for the pipeline including augmentation (`aug.py`), data importing (`import_data.py`), image preprocessing (`image_preprocessing.py`), and tiling (`tile_image.py`).
*   **`Model_Training/`**: Contains `Model.ipynb` for the core model training workflow.
*   **`models_a/`**: Saved weights for the refinement models (e.g., `best_refiner_model.joblib`, `ridge_model.joblib`).
*   **`predictions/`**: Out-of-fold predictions and ground truths stored as `.npy` arrays, alongside final model predictions.
*   **Notebooks (`*.ipynb`)**: Root-level notebooks (`Cell_classification.ipynb`, `BM616projecttrial1.ipynb`) used for initial trials, EDA, and classification pipelines.

## Methodology

### 1. Base Models
The baseline is a **ResNet18** architecture that takes a fixed-size patch and directly outputs a 35-dimensional prediction vector using an L1 loss function. To enhance this, we implemented a **Multi-scale CNN** that processes three inputs: a central patch, a grid of surrounding sub-patches, and a broader context tile.

### 2. Two-Stage Refinement
Because biological tissue exhibits structured relationships, predicting cell types independently leads to inconsistencies. We introduced an **ElasticNet regression** refinement stage. It learns a linear combination of all predicted cell types to enforce global consistency:
$\hat{y}_{i}=\sum_{j=1}^{35}w_{ij}x_{j}$
where $x_{j}$ are the first-stage predictions and $w_{ij}$ are the learned coefficients[cite: 1]. To prevent data leakage, this second stage is trained strictly on out-of-fold (OOF) predictions.

*(Note: We also explored a Graph Convolutional Network (GCN) based on spatial proximity, but it did not outperform the CNN baseline.)*

## Results

Performance was measured using the mean Spearman correlation across all spatial locations via Leave-One-Slide-Out Cross-Validation (LOSOCV).

*   **ResNet18 (Base):** ~0.45
*   **Multi-scale CNN:** 0.6387
*   **ResNet18 + ElasticNet Refinement:** ~0.62
*   **ResNet18 + ElasticNet + Quantile Post-Processing:** ~0.64

Our experiments prove that while improving the base model aids feature extraction, explicitly modeling the sparse, inter-cell dependencies yields the most significant performance gains, pushing our approach into the competitive range of the Global AI Hackathon 25 benchmarks.
