from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(batch_size=32):
    """
    Loads the Adinkra dataset, splits it, and returns DataLoaders.
    """
    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "raw"

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

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    base_dataset = datasets.ImageFolder(root=str(dataset_path))

    total_size = len(base_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_full = datasets.ImageFolder(root=str(dataset_path), transform=train_transform)
    val_full = datasets.ImageFolder(root=str(dataset_path), transform=test_transform)
    test_full = datasets.ImageFolder(root=str(dataset_path), transform=test_transform)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    test_dataset = Subset(test_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n--- Data Pipeline Ready ---")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_loader, val_loader