import torch
from torchvision import transforms
from PIL import Image
from adinkra_cnn import AdinkraCNN


def load_model(num_classes=96, model_path="best_adinkra_model (81).pth", device="cpu"):
    """
    Loads the trained Adinkra CNN model dynamically based on class count.
    """
    # Initialize model with the dynamic class count
    model = AdinkraCNN(num_classes=num_classes)

    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # turns off dropout
    model.eval()
    model.to(device)

    return model


def predict_image(model, image_path, device="cpu"):
    """
    Takes a model and an image path, and returns the predicted class ID.
    """
    #  Load the image
    image = Image.open(image_path).convert("RGB")

    #  transform
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        output = model(input_tensor)

        _, predicted_idx = torch.max(output, 1)

    return predicted_idx.item()
