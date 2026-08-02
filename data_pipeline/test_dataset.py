"""
test_dataset.py

Purpose:
--------
This file tests that our custom AdinkraDataset works correctly.

It also verifies:
1. The dataset loads successfully.
2. The dataset is split into training, validation, and testing sets.
3. DataLoaders create batches correctly.
4. Images and labels have the expected shape.
"""

from dataset import AdinkraDataset
from torch.utils.data import DataLoader, random_split


# ==========================================================
# STEP 1: Load the processed Adinkra dataset
# ==========================================================

dataset = AdinkraDataset(
    r"C:\Users\Vannessa Brose\OneDrive\Desktop\OpenCV\processed_data"
)

print(f"Total number of images: {len(dataset)}")


# ==========================================================
# STEP 2: Split the dataset
# ----------------------------------------------------------
# 70% -> Training
# 15% -> Validation
# 15% -> Testing
# ==========================================================

dataset_size = len(dataset)

train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)

# The remaining images go into the test set
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print("\nDataset Split")
print("---------------------")
print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Testing Images    : {len(test_dataset)}")


# ==========================================================
# STEP 3: Create DataLoaders
# ----------------------------------------------------------
# batch_size = Number of images sent to the CNN at once.
#
# shuffle=True  -> Randomize training images
# shuffle=False -> Keep validation/testing order unchanged
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================================
# STEP 4: Test one training batch
# ----------------------------------------------------------
# Display:
# - Batch shape
# - Labels inside the batch
# ==========================================================

print("\nFirst Training Batch")
print("---------------------")

for images, labels in train_loader:

    print("Batch Shape:", images.shape)
    print("Labels:", labels)

    # Only display the first batch
    break


# ==========================================================
# STEP 5: Display the number of batches
# ----------------------------------------------------------
# This is NOT the number of images.
# It is the number of batches created by the DataLoader.
# ==========================================================

print("\nNumber of Batches")
print("---------------------")
print("Training Batches  :", len(train_loader))
print("Validation Batches:", len(val_loader))
print("Testing Batches   :", len(test_loader))


# ==========================================================
# STEP 6: Test loading a single image
# ----------------------------------------------------------
# The Dataset should return:
# - One image tensor
# - One numerical label
# ==========================================================

print("\nSingle Image Test")
print("---------------------")

image, label = dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)