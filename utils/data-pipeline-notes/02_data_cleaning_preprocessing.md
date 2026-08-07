# Stage 2: Data Cleaning and Image Preprocessing

## Objective

The objective of this stage is to clean and standardize the Adinkra image
dataset so that the images can be used consistently during CNN training.

## Problems Identified During Dataset Inspection

The dataset inspection revealed the following:

- The dataset contains 785 images.
- The images belong to 10 Adinkra symbol classes.
- Images are stored in JPEG and PNG formats.
- Image dimensions are inconsistent.
- Some images are extremely small.
- The dataset is imbalanced across the 10 classes.
- No unreadable images were detected.

## Preprocessing Strategy

The preprocessing pipeline will perform the following operations:

1. Validate and read images using OpenCV.
2. Preserve the original images in the raw dataset.
3. Resize images to a consistent CNN input size.
4. Preserve image aspect ratio during resizing.
5. Use padding where necessary to prevent distortion.
6. Convert images to a consistent format where necessary.
7. Save the processed images separately from the raw dataset.

## Target Image Size

The target image size is:

**224 × 224 pixels**

This provides a consistent input size for the CNN.

## Data Augmentation

Data augmentation will be performed during model training rather than
permanently modifying the raw or processed dataset.

Planned augmentations include:

- Minor rotations
- Small scaling variations

Horizontal and vertical flipping will not be used because some Adinkra
symbols are asymmetric and flipping them may change their visual
characteristics.

## Class Imbalance

The dataset is imbalanced, with Sankofa having the highest number of images
and Dame Dame having the lowest.

The appropriate strategy for addressing class imbalance will be evaluated
during model development.

## Results

To be completed after preprocessing.

## Problems Encountered

To be completed during implementation.

## Conclusion

The preprocessing pipeline will standardize the images while preserving
important visual characteristics of the Adinkra symbols.


## Data Cleaning

During the inspection of very small images, two 16 × 16 images were
identified in the Aya class:

- `shop.paulsarz.png`
- `yvettemichele.png`

These files were removed after inspection because they were not suitable
Adinkra symbol training images.

The remaining small images were retained because they belong to the
corresponding Adinkra classes and may still contain useful visual
information.

### Cleaning Summary

- Original dataset: 785 images
- Images removed: 2
- Current dataset: 783 images
- Reason for removal: irrelevant/non-Adinkra images
- Raw dataset was otherwise preserved.

2. # Adinkra AI — Dataset Preprocessing

## 1. Overview

The preprocessing stage prepares the raw Adinkra symbol images for use by the Convolutional Neural Network (CNN).

The raw dataset contains images of **101 different Adinkra symbol classes**. The images come in different formats, dimensions, and aspect ratios. A CNN requires the input images to have a consistent size and format.

Therefore, the preprocessing pipeline performs the following operations:

1. Reads images from the raw dataset.
2. Identifies the 101 symbol classes.
3. Validates supported image formats.
4. Reads images using OpenCV.
5. Resizes images to **224 × 224 pixels**.
6. Preserves the original image aspect ratio.
7. Adds white padding where necessary.
8. Saves the processed images in a structured directory.
9. Converts all processed images to `.jpg`.
10. Reports successful and failed preprocessing operations.

---

# 2. Dataset Structure

The raw dataset follows a folder-based classification structure.

Each folder represents one Adinkra symbol class.

```text
dataset/
│
├── raw/
│   ├── Aban/
│   │   ├── image1.jpeg
│   │   ├── image2.png
│   │   └── ...
│   │
│   ├── Abe_Dua/
│   │   ├── image1.jpeg
│   │   └── ...
│   │
│   ├── Adinkrahene/
│   │   └── ...
│   │
│   └── ...
│
└── processed/

Data augmentation was implemented at training time using PyTorch's torchvision.transforms. Random rotation, horizontal flipping, and random cropping are applied to training images to increase variation and improve the model's ability to generalize. Validation and testing images are not randomly augmented to ensure fair evaluation.