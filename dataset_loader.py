import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset




DATASET_PATH = "dataset/processed"



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available. Using CPU.")




train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Randomly rotate the image by up to +/- 15 degrees
    transforms.RandomRotation(15),

    # Randomly flip the image horizontally
    transforms.RandomHorizontalFlip(),

    # Randomly crop the image while allowing a small padding
    transforms.RandomCrop(224, padding=10),

    # Convert image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize image using ImageNet mean and standard deviation
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Convert image to PyTorch tensor
    transforms.ToTensor(),

    # Normalize image
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




base_dataset = datasets.ImageFolder(
    root=DATASET_PATH
)

print("\nTotal images:", len(base_dataset))
print("Classes:", base_dataset.classes)
print("Class mapping:", base_dataset.class_to_idx)




total_size = len(base_dataset)

# 70% training
train_size = int(0.70 * total_size)

# 15% validation
val_size = int(0.15 * total_size)

# Remaining 15% testing
test_size = total_size - train_size - val_size




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



# ------------------------------------------------------------

train_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=train_transform
)




val_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=eval_transform
)




test_full = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=eval_transform
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



print("\nDataset Split")
print("----------------")

print("Training images  :", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images   :", len(test_dataset))




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




images, labels = next(iter(train_loader))

print("\nFirst Training Batch")
print("---------------------")

print("Batch shape:", images.shape)
print("Labels:", labels)




images = images.to(device)
labels = labels.to(device)

print("\nDevice Test")
print("---------------------")

print("Images device:", images.device)
print("Labels device:", labels.device)




print("\nNumber of Batches")
print("---------------------")

print("Training batches  :", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches   :", len(test_loader))




print("\nPipeline Validation")
print("---------------------")


# Check image batch shape
if images.shape == (32, 3, 224, 224):
    print("✓ Image batch shape is correct")
else:
    print("✗ Image batch shape is incorrect")


# Check total dataset size
if len(base_dataset) == 5089:
    print("✓ Dataset contains 5089 images")
else:
    print(
        "WARNING: Dataset contains",
        len(base_dataset),
        "images"
    )


# Check number of classes
if len(base_dataset.classes) == 101:
    print("✓ 101 classes detected")
else:
    print("✗ Incorrect number of classes")


# Check device
if device.type == "cuda":
    print("✓ Batch successfully moved to GPU")
else:
    print("⚠ GPU not available; batch is running on CPU")
