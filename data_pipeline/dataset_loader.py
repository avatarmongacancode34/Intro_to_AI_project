import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_dataloaders(data_dir, batch_size=32, random_seed=42):
    """
    Loads images from data_dir, applies stratified 70/15/15 splitting, 
    and returns DataLoaders with appropriate transformations.
    """
    
    # Transformations
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=10), # Note: Keep an eye on border artifacts here!
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    
    #  dataset loading
    
    base_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(base_dataset.classes)
    class_names = base_dataset.classes

    
    # stratified splitting to avoid having small class in one set
    
    targets = base_dataset.targets
    indices = list(range(len(base_dataset)))

    train_indices, temp_indices, train_targets, temp_targets = train_test_split(
        indices, 
        targets, 
        train_size=0.70, 
        stratify=targets, 
        random_state=random_seed
    )

    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=0.50,
        stratify=temp_targets, 
        random_state=random_seed
    )

   
    # tranfrom datasets
    
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_full = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_full = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    test_dataset = Subset(test_full, test_indices)

    
    # dataloaders
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, class_names




if __name__ == "__main__":
    DATASET_PATH = "dataset/processed"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nInitializing DataLoaders...")
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=DATASET_PATH, 
        batch_size=32
    )

    print(f"\nDetected {num_classes} classes.")
    print(f"Training batches  : {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches   : {len(test_loader)}")

    # Test a single batch
    images, labels = next(iter(train_loader))
    images = images.to(device)
    
    print("\nPipeline Validation:")
    if images.shape == (32, 3, 224, 224):
        print("✓ Image batch shape is correct")
    else:
        print("✗ Image batch shape is incorrect")
