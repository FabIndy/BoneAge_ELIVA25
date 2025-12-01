# Model Weights — Bone Age Estimation (Segmentation + V13 Regression)

This directory contains *references* to the trained model weights used in the
BoneAge / ELIVA25 pipeline.  
**The actual `.keras` files are NOT stored in this repository** because GitHub
does not allow large binary files (>100 MB).

Instead, all trained weights are hosted externally (Google Drive).

---

## Overview of Available Models

### 1.  `seg_model_best.keras`  
**Task:** Hand segmentation (binary mask)  
**Architecture:** ResNet50 encoder + UNet decoder  
**Training notebook:** `01_segmentation_training.ipynb`  
**Input:** `(224, 224, 3)`  
**Output:** `(224, 224, 1)` sigmoid mask  
**Dataset:** Cleaned RSNA radiographs (mask annotations)

This model is used to automatically generate hand masks on ELIVA25 and RSNA images.

**Download link:**  
*https://drive.google.com/drive/folders/16LPLfu9FusRrXYWeLGV2uFCVtOHFtZWY?usp=sharing*  



---

### 2.  `model_v13_preproc_v3.keras`  
**Task:** Bone age regression  
**Architecture:** EfficientNetV2B3 (mixed precision, regression head)  
**Training notebook:** `05_training_v13.ipynb`  
**Input:** `(300, 300, 3)` image + sex indicator  
**Output:** bone age (in months)  
**Preprocessing:**  
- segmentation mask applied  
- CLAHE enhancement  
- resize to 300×300  
- `preprocess_input` (EfficientNetV2)

This is the model used to produce the final predictions for
**ELIVA25 test set** in the notebook  
`10_inference_test_eliva25_v13.ipynb`.

**Download link:**  
*https://drive.google.com/drive/folders/16LPLfu9FusRrXYWeLGV2uFCVtOHFtZWY?usp=sharing*  



---



