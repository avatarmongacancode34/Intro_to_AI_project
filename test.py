import torch

# 1. Prepare the Device and Architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (Assuming your architecture class is named AdinkraCNN - adjust if yours is named differently)
# model = AdinkraCNN(num_classes=96).to(device) 

# Load the exact weights from your 81.03% run
model.load_state_dict(torch.load('best_adinkra_model.pth (81)', map_location=device))

# 2. Safety Lock 1: Disable training layers like Dropout
model.eval()

# Variables to keep score
correct_predictions = 0
total_predictions = 0

# 3. Safety Lock 2: Disable gradient calculation to save memory and speed up testing
with torch.no_grad():
    
    # 4. The Evaluation Loop
    for images, labels in test_loader: # Ensure this is your test_loader, not train_loader!
        
        # Move the test batch to the GPU
        images, labels = images.to(device), labels.to(device)

        # Run the images through the network
        outputs = model(images)

        # Grab the class index with the highest probability score
        _, predicted = torch.max(outputs.data, 1)

        # Tally up the totals for this batch
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

# Calculate the final percentage
test_accuracy = 100 * correct_predictions / total_predictions
print(f'Final Test Set Accuracy: {test_accuracy:.2f}%')
