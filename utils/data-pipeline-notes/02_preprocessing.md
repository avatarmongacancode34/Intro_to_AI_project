# Stage 2: Data Preprocessing

## Objective

The objective of this stage was to preprocess the cleaned Adinkra symbol dataset
into a consistent format suitable for training a computer vision model.

## Input Dataset

The cleaned dataset contained 783 images across 10 Adinkra symbol classes.

The classes were:

- Abe_Dua
- Adinkrahene
- Agyinduwura
- Aya
- Bese_Saka
- Dame_Dame
- Denkyem
- Dwannimmen
- Nyame_Akruma
- Sankofa

## Preprocessing Operations

The preprocessing pipeline was applied to every image in the dataset.

The pipeline standardized the image data so that the images could be used
consistently by the machine learning model.


## Data Duplication and Filename Collision Issue

During preprocessing, a filename collision issue was identified in the dataset.

There were 63 pairs of JPEG and PNG files that shared identical filename stems.
For example:

- `1683619858536.jpeg`
- `1683619858536.png`

Although these were different files, the initial preprocessing pipeline converted
both images to `.jpg` while retaining the same filename stem.

This caused one processed image to overwrite the other.

### Impact

During the initial preprocessing attempt:

- Raw images: 783
- Successfully processed images: 783
- Images remaining after preprocessing: 720

The difference of 63 images was caused by filename collisions during conversion.

### Solution

The preprocessing pipeline was modified so that the original file format is
included in the output filename.

This prevents JPEG and PNG files with the same filename stem from overwriting
each other.

For example, instead of producing:

`1683619858536.jpg`

the pipeline preserves the distinction between the source files when generating
the processed filenames.

### Verification

After modifying the preprocessing pipeline, the processed dataset was deleted
and preprocessing was run again from the beginning.

The final validation produced:

- Total processed images: 783
- Incorrect image sizes: 0
- Unreadable images: 0
- Processing failures: 0

The validation status was:

**PREPROCESSING VALIDATION PASSED**

This confirmed that the filename collision issue had been resolved and all valid
images were successfully preserved.

## Preprocessing Results

Total images processed: 783

Successfully processed: 783

Failed: 0

Incorrect image sizes: 0

Unreadable images: 0

## Validation

A separate validation script was used to verify the preprocessing results.

The validation confirmed that:

- All 783 images were successfully processed.
- All images had the required dimensions.
- No processed images were unreadable.
- The number of images in each class remained consistent with the cleaned dataset.

## Final Class Distribution

| Class | Images |
|---|---:|
| Abe_Dua | 61 |
| Adinkrahene | 56 |
| Agyinduwura | 54 |
| Aya | 92 |
| Bese_Saka | 96 |
| Dame_Dame | 39 |
| Denkyem | 48 |
| Dwannimmen | 106 |
| Nyame_Akruma | 44 |
| Sankofa | 187 |
| Total | 783 |

## Validation Status

PREPROCESSING VALIDATION PASSED

## Conclusion

The preprocessing stage was successfully completed. The resulting dataset
contains 783 valid, consistently processed images across 10 Adinkra symbol
classes and is ready for the next stage of the machine learning pipeline.

