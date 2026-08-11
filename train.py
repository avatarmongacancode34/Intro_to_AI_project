import torch
import torch.nn as nn
import torch.optim as optim
import time

# Import our custom modules
from data_pipeline.dataset_loader import get_dataloaders
from adinkra_cnn import AdinkraCNN

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "dataset/raw"
BATCH_SIZE = 32
NUM_EPOCHS = 120
LEARNING_RATE = 0.001
SAVE_PATH = "best_adinkra_model.pth"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=DATASET_PATH,
        batch_size=BATCH_SIZE
    )
    print(f"Detected {num_classes} classes.")

    # 2. Initialize Model, Loss, and Optimizer
    model = AdinkraCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0

    # 3. Training Loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - start_time

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {epoch_duration:.0f}s "
              f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% "
              f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 4. Save the Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  => Validation accuracy improved. Saved model to {SAVE_PATH}")
        scheduler.step(val_acc)

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()


