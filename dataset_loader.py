import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset



# PATH


DATASET_PATH = "dataset/processed"



# DEVICE


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available. Using CPU.")



# TRANSFORMATIONS


# Training transformations
# Data augmentation is applied ONLY during training.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and testing transformations
# No random augmentation is applied.

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# LOAD BASE DATASET


base_dataset = datasets.ImageFolder(
    root=DATASET_PATH
)

print("\nTotal images:", len(base_dataset))
print("Classes:", base_dataset.classes)
print("Class mapping:", base_dataset.class_to_idx)



# CREATE TRAIN / VALIDATION / TEST SPLIT


total_size = len(base_dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size


# Generate reproducible indices
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    total_size,
    generator=generator
).tolist()


train_indices = indices[:train_size]

val_indices = indices[
    train_size:train_size + val_size
]

test_indices = indices[
    train_size + val_size:
]



# CREATE DATASETS


# Training dataset
train_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=train_transform
)

# Validation dataset
val_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=test_transform
)

# Testing dataset
test_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=test_transform
)


# Apply the same split indices to each dataset
train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)

test_dataset = Subset(
    test_full,
    test_indices
)



# DATASET SPLIT INFORMATION


print("\nDataset Split")
print("----------------")
print("Training images  :", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images   :", len(test_dataset))



# CREATE DATALOADERS


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



# TEST TRAINING BATCH


images, labels = next(iter(train_loader))


print("\nFirst Training Batch")
print("---------------------")
print("Batch shape:", images.shape)
print("Labels:", labels)



# MOVE BATCH TO GPU


images = images.to(device)
labels = labels.to(device)


print("\nGPU Test")
print("---------------------")
print("Images device:", images.device)
print("Labels device:", labels.device)



# NUMBER OF BATCHES


print("\nNumber of Batches")
print("---------------------")
print("Training batches  :", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches   :", len(test_loader))



# FINAL VALIDATION


print("\nPipeline Validation")
print("---------------------")

if images.shape == (32, 3, 224, 224):
    print("✓ Image batch shape is correct")
else:
    print("✗ Image batch shape is incorrect")

if len(base_dataset) == 783:
    print("✓ Dataset contains 783 images")
else:
    print("WARNING: Dataset contains", len(base_dataset), "images")

if len(base_dataset.classes) == 10:
    print("✓ 10 classes detected")
else:
    print("✗ Incorrect number of classes")

if device.type == "cuda":
    print("✓ Batch successfully moved to GPU")
else:
    print("⚠ GPU not available; batch is running on CPU")