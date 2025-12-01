# Bone Age Estimation — RSNA Dataset + ELIVA25 Kaggle Challenge

This repository contains a complete end-to-end pipeline for **automatic bone age estimation**
using deep learning on hand radiographs.

The project combines:

- **Hand segmentation** using a custom ResNet-UNet  
- **Preprocessing and domain adaptation**  
- **EfficientNetV2-based regression model (V13)**  
- **Full inference pipeline** on the *ELIVA25* Kaggle test set  
- **Kaggle-ready submission file**

It is designed as both:
- a **realistic applied ML project** (RSNA → preprocessing → modeling → inference)
- a **Kaggle-style experiment** with multiple model variants and ablation studies.

---

#  Project Structure

```
Boneage/
│
├── models/                     # external model weights (README only)
│   ├── README.md
│
├── notebooks/                  #  11 notebooks
│
├── data_samples/               # 3 anonymized sample images
│
├── submission_v13.csv          # final Kaggle submission (MAE = 20.72)
│
├── requirements.txt
└── README_repo_boneage.md                   # (this file)
```

*(No dataset or `.keras` files are stored in this repository — GitHub size limits.)*

---

# Objective

The goal is to estimate **bone age (in months)** from left-hand X-ray images.

The training dataset is the **RSNA Bone Age dataset**, while the final inference is performed on the **ELIVA25** Kaggle challenge (33 unlabeled test images).

This project explores:

- segmentation vs non-segmentation pipelines  
- preprocessing strategies (CLAHE, resizing, normalization)  
- domain shift between RSNA and ELIVA25  
- EfficientNetV2 architectures  
- robust deep learning on tiny unlabeled test sets  

---

# 1. Hand Segmentation (UNet + ResNet50 encoder)

**Input:** Raw radiograph  
**Output:** Binary hand mask

Pipeline:

1. Resize to 224×224  
2. Apply UNet segmentation  
3. Threshold mask (0.5 → 0/1)  
4. Apply mask to original-resolution image  
5. Export segmented test images

This step improves robustness and reduces background noise.

---

# 2. Preprocessing Pipeline

All images (RSNA + ELIVA25) follow the same logic:

- grayscale  
- segmentation mask applied  
- CLAHE enhancement  
- resize to 300×300  
- EfficientNetV2 `preprocess_input`  
- sex metadata encoded as float32 (0 or 1)

Output folder for ELIVA25 test images:

```
preprocessed_test_eliva25/
```

---

# 3. Regression Model — V13 (EfficientNetV2B3)

Architecture:

- EfficientNetV2B3 backbone  
- global average pooling  
- fully-connected regression head  
- mixed precision (float16)  
- L2 regularization  
- MSE loss, MAE tracking  
- input: image + sex indicator  

Training notebook:  
`05_training_v13.ipynb`

---

# 4. Inference on ELIVA25 Test Set

Notebook:  
`10_inference_test_eliva25_v13.ipynb`

Steps:

1. Load preprocessed images  
2. Build TensorFlow dataset  
3. Apply the exact preprocessing as in training  
4. Predict bone age (months)  
5. Create `submission_v13.csv`  

**Final public leaderboard score:**  
**MAE = 20.72**

Comparison:

| Model version | Pipeline                         | MAE       |
|---------------|----------------------------------|-----------|
| V9 unmasked   | no segmentation                  | 23.05     |
| V10 unmasked  | stronger augmentations           | 24.03     |
| **V13 masked**| **segmentation + preprocessing** | **20.72** |

---

#  What I Learned

This project provided hands-on experience with:

### Domain shift  
RSNA → ELIVA25 (exposure, crop, noise, resolution)

### Impact of segmentation  
Removing background improves regression stability.

### Consistent preprocessing  
Training and inference must use *identical* pipelines.

### Tiny test set inference  
Only 33 images → extremely careful pipeline required.

### EfficientNetV2 architectures  
Strong and stable for medical imaging regression.

### Reproducibility  
Clear notebook order, external weight hosting, controlled randomness.

This project illustrates realistic medical imaging workflows, and also the specific constraints of a Kaggle-style challenge.

---

# Installation & Reproduction

Install dependencies:

```
pip install -r requirements.txt
```

Download weights from the links in `models/README.md`.

Run notebooks in the order.


---

# Acknowledgements

- RSNA Pediatric Bone Age Dataset  
- ELIVA25 Kaggle Challenge  
- EfficientNetV2  
- TensorFlow Mixed Precision  

---

#  Author

**Fabrice Belfiore**  
Based in Indianapolis, IN  
Data Scientist — *training completed, actively seeking opportunities*
