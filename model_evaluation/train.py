import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from adinkra_cnn import AdinkraCNN
from evaluation import evaluate_model
import time


def collect_predictions(model, data_loader, device):
  model.eval()
  y_true = []
  y_pred = []

  with torch.no_grad():
    for images, labels in data_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs, 1)

      y_true.extend(labels.cpu().tolist())
      y_pred.extend(predicted.cpu().tolist())

  return y_true, y_pred

def train_model(model, train_loader,val_loader,criterion,optimizer, num_epochs,device):
  print(f"Starting training for {num_epochs} epochs on {device}\n")

  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    start_time = time.time()

    # train
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
      images = images.to(device)
      labels =  labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

      # tracking stats

      running_train_loss += loss.item() * images.size(0)
      _, predicted = torch.max(outputs.data, 1)
      total_train += labels.size(0)
      correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = 100 * correct_train / total_train

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        running_val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = 100 * correct_val / total_val

    #saving best model
    if epoch_val_loss < best_val_loss:
      best_val_loss = epoch_val_loss
      torch.save(model.state_dict(), 'best_adinkra_model.pth')
      saved_msg = "Model saved"
    else:
      saved_msg = "Model not saved"
    end_time = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {end_time - start_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}{saved_msg}")
              
  print("\nTraining complete. Best weights saved to 'best_adinkra_model.pth'.")
  return model
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  NUM_CLASSES = 10
  NUM_EPOCHS = 5
  BATCH_SIZE = 16
  LEARNING_RATE = 0.001

  from data_pipeline.dataset_loader import get_dataloaders
  train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

  model = AdinkraCNN(num_classes=NUM_CLASSES).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  trained_model = train_model(model, train_loader,val_loader,criterion,optimizer, NUM_EPOCHS,device)

  best_model_path = "best_adinkra_model.pth"
  trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
  y_true, y_pred = collect_predictions(trained_model, val_loader, device)
  evaluate_model(y_true, y_pred)