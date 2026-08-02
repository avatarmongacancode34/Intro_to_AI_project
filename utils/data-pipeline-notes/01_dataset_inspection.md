# Stage 1: Dataset Inspection

## Objective

The objective of this stage is to inspect and understand the Adinkra image
dataset before preprocessing and model training.

## Dataset Organization

The dataset contains images organized into separate folders, where each
folder represents an Adinkra symbol class.

The current dataset contains 10 symbol classes:

1. Abe Dua
2. Adinkrahene
3. Agyinduwura
4. Aya
5. Bese Saka
6. Dame Dame
7. Denkyem
8. Dwannimmen
9. Nyame Akuma
10. Sankofa

## Dataset Structure

```text
dataset/
└── raw/
    ├── Abe_Dua/
    ├── Adinkrahene/
    ├── Agyinduwura/
    ├── Aya/
    ├── Bese_Saka/
    ├── Dame_Dame/
    ├── Denkyem/
    ├── Dwannimmen/
    ├── Nyame_Akuma/
    └── Sankofa/

## Results

The initial inspection identified 10 Adinkra symbol classes containing a
total of 785 images.

### Class Distribution

| Adinkra Symbol | Number of Images |
|---|---:|
| Abe Dua | 61 |
| Adinkrahene | 56 |
| Agyinduwura | 54 |
| Aya | 94 |
| Bese Saka | 96 |
| Dame Dame | 39 |
| Denkyem | 48 |
| Dwannimmen | 106 |
| Nyame Akruma | 44 |
| Sankofa | 187 |
| **Total** | **785** |

### Initial Observation

The dataset is imbalanced across the 10 classes.

Sankofa has the largest number of images (187), while Dame Dame has the
smallest number of images (39). Therefore, the dataset will require further
analysis before model training to determine how class imbalance should be
handled.

No images were removed at this stage. The original dataset has been
preserved in the `dataset/raw/` directory.

## Conclusion

The dataset contains 785 images across 10 Adinkra symbol classes. The next
stage of data inspection will examine image formats, image dimensions,
readability, and potentially corrupted images before preprocessing.

### Image Format Distribution

The dataset contains two supported image formats:

| Format | Number of Images |
|---|---:|
| JPEG | 363 |
| PNG | 422 |
| **Total** | **785** |

### Image Dimension Analysis

The images do not have a consistent resolution. Multiple image dimensions
were identified, ranging from very small images such as 16 × 16 pixels to
larger images such as 471 × 107 pixels.

Some of the most common dimensions were:

| Dimensions | Number of Images |
|---|---:|
| 225 × 225 | 279 |
| 224 × 224 | 12 |
| 216 × 233 | 25 |
| 223 × 226 | 25 |
| 194 × 259 | 24 |
| 275 × 183 | 24 |

This variation means that image resizing will be required during the
preprocessing stage so that all images have a consistent input size for the
CNN.

### Image Readability

OpenCV was used to attempt to read all supported images.

**Unreadable images: 0**

Therefore, no corrupted or unreadable images were identified during this
initial inspection.

### Dataset Imbalance

The dataset is not evenly distributed across the ten classes. Sankofa has
the largest number of images (187), while Dame Dame has the smallest (39).

This imbalance will be considered during the model development stage to
reduce the possibility of the CNN becoming biased toward classes with more
training examples.

##
Dataset Summary
---------------
Total images: 785

Classes: 10

Formats:
.jpeg: 363
.png: 422

Most Common Dimensions:
225x225: 279
216x233: 25
223x226: 25
...

Unreadable Images: 0

Class Imbalance:
Largest: Sankofa (187)
Smallest: Dame_Dame (39)