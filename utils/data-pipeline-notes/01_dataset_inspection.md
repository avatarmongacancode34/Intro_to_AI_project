# Adinkra AI — Dataset Inspection

## Stage 3: Dataset Inspection and Quality Assessment

### Purpose

The purpose of this stage is to inspect the raw Adinkra dataset before preprocessing and model training.

The inspection checks:

- Number of Adinkra symbol classes
- Number of images per class
- Image file formats
- Image dimensions
- Very small images
- Corrupted or unreadable images
- Filename collisions
- Class imbalance

This helps us identify potential problems that could affect the performance of the CNN model.

---

## 1. Dataset Overview

The dataset contains **101 Adinkra symbol classes**.

| Property | Result |
|---|---:|
| Number of classes | 101 |
| Total images | 5,089 |
| Unreadable images | 0 |
| Very small images | 249 |
| Filename collisions | 479 |
| Image formats | JPEG, PNG, JPG |

The raw dataset is located at:

```text
dataset/raw/