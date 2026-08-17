import torch

from adinkra_cnn import AdinkraCNN
from data_pipeline.dataset_loader import get_dataloaders


dataset_path = '/content/Intro_to_AI_project/dataset/raw' 


_, _, test_loader, num_classes, _ = get_dataloaders(data_dir=dataset_path, batch_size=32)

print(f"Successfully loaded test_loader with {len(test_loader)} batches!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AdinkraCNN(num_classes=96).to(device)


model.load_state_dict(torch.load('/content/Intro_to_AI_project/best_adinkra_model (81).pth', map_location=device))


model.eval()

correct_predictions = 0
total_predictions = 0


with torch.no_grad():
    
    
    for images, labels in test_loader: 
        
       
        images, labels = images.to(device), labels.to(device)

        
        outputs = model(images)

        
        _, predicted = torch.max(outputs.data, 1)

       
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()


test_accuracy = 100 * correct_predictions / total_predictions
print(f'Final Test Set Accuracy: {test_accuracy:.2f}%')
