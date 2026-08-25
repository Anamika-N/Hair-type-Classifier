# Hair Type Classification using VGG16 & ViT

> **Self Project** | TensorFlow · Keras · HuggingFace Transformers · Transfer Learning · Computer Vision

---

## Overview

An image classification pipeline that identifies 5 hair types from photographs, comparing two transfer learning approaches — a fine-tuned **VGG16** CNN and a fine-tuned **Vision Transformer (ViT)**. The project covers the full ML workflow: data preparation, augmentation, class balancing, model training, and evaluation.

---

## Hair Type Classes

| Label | Class |
|---|---|
| 0 | Straight |
| 1 | Wavy |
| 2 | Curly |
| 3 | Kinky |
| 4 | Dreadlocks |

---

## Dataset

- ~2,000 images across 5 hair type classes
- Source: [Hair Type Dataset (Kaggle)](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset)
- Preprocessing: resize, normalize to `[0, 1]`, train/validation/test splits
- Class balancing via `RandomOverSampler` (ViT pipeline)
- Data augmentation: random height/width shift, zoom, brightness (VGG16); random rotation, sharpness, horizontal flip (ViT)

---

## Model 1 — VGG16 (TensorFlow / Keras)

**Architecture:**
```
VGG16 (ImageNet weights, top removed)
  Last 4 layers unfrozen for fine-tuning
    → Dense(128, relu)
    → Conv2D(128, 3×3, relu)
    → Dropout(0.5)
    → GlobalAveragePooling2D
    → Flatten
    → Dense(5, softmax)
```

**Training config:**
- Input size: `128×128` (train), `224×224` (validation)
- Optimizer: Adam, `lr = 1e-5`
- Loss: Categorical Crossentropy
- Epochs: 40, Batch size: 32
- Best weights saved via `ModelCheckpoint`

---

## Model 2 — Vision Transformer ViT (HuggingFace Transformers + PyTorch)

**Pretrained model:** `google/vit-base-patch16-224-in21k`

**Architecture:**
```
ViT-Base/16 (pretrained on ImageNet-21k)
    → Classification head (num_labels=5)
```

**Training config:**
- Input size: `224×224`
- Learning rate: `1e-6`
- Epochs: 100
- Augmentation: random rotation, sharpness, horizontal flip
- Evaluation metric: accuracy, macro-F1, confusion matrix

---

## Project Structure

```
hair-type-classification-vgg16-vit/
├── hair-style-types-recognition-using-vgg16.ipynb   # VGG16 pipeline
├── hair-type-image-detection-vit.ipynb              # ViT pipeline
├── requirements.txt
└── README.md
```

### Dataset structure expected
```
data/
├── straight/
├── wavy/
├── curly/
├── kinky/
└── dreadlocks/
```

---

## Setup

```bash
pip install -r requirements.txt
```

Open either notebook in Jupyter or Kaggle. The notebooks use `/kaggle/input/hair-type-dataset/data` as the data path — update this to your local path if running outside Kaggle.
