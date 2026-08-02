# Stage 3: Dataset Loading and Splitting

## Objective

The objective of this stage was to load the preprocessed Adinkra image dataset into a format suitable for training a deep learning model.

## Dataset Loading

The dataset loader reads the preprocessed images from the dataset directory and automatically assigns numerical labels to the ten Adinkra symbol classes.

### Classes

1. Abe_Dua
2. Adinkrahene
3. Agyinduwura
4. Aya
5. Bese_Saka
6. Dame_Dame
7. Denkyem
8. Dwannimmen
9. Nyame_Akruma
10. Sankofa

## Class Mapping

| Class | Label |
|---|---:|
| Abe_Dua | 0 |
| Adinkrahene | 1 |
| Agyinduwura | 2 |
| Aya | 3 |
| Bese_Saka | 4 |
| Dame_Dame | 5 |
| Denkyem | 6 |
| Dwannimmen | 7 |
| Nyame_Akruma | 8 |
| Sankofa | 9 |

## Dataset Split

The 783 images were divided into training, validation, and testing sets.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 548 | 70% |
| Validation | 117 | 15% |
| Testing | 118 | 15% |
| **Total** | **783** | **100%** |

## Image Representation

Images were loaded as RGB images and resized to 224 × 224 pixels.

The resulting tensor shape for a batch was:

`[32, 3, 224, 224]`

where:

- 32 = batch size
- 3 = RGB colour channels
- 224 × 224 = image dimensions

## DataLoaders

The datasets were loaded into PyTorch DataLoaders using a batch size of 32.

| Dataset | Number of Batches |
|---|---:|
| Training | 18 |
| Validation | 4 |
| Testing | 4 |

## Validation

The dataset loader was executed successfully.

Results:

- Total images loaded: 783
- All 10 classes were detected.
- Dataset splitting completed successfully.
- The first training batch was successfully loaded.
- Batch dimensions were verified as 32 × 3 × 224 × 224.
- Training, validation, and testing DataLoaders were successfully created.

## Conclusion

Stage 3 was successfully completed. The preprocessed dataset is now loaded, labelled, split into training/validation/testing sets, and prepared for CNN model training.