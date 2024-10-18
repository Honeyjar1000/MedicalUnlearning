import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
import medmnist
from medmnist import INFO

# Configuration
DATASET_NAME = "chestmnist"
BATCH_SIZE = 8  # Adjust this depending on how many samples you want to visualize
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Dataloader
info = INFO[DATASET_NAME]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the test dataset
test_dataset = DataClass(split='test', transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the trained ResNet-18 model
model = models.resnet18(pretrained=False)

# Modify the first convolutional layer to accept 1 input channel instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 14)  # Adjust for 14 output classes

# Load the model weights
model.load_state_dict(torch.load("resnet18_chestmnist.pth"))
model = model.to(DEVICE)
model.eval()

# Function to display images along with ground truth and predictions
def display_sample_images(model, device, test_loader, num_images=8):
    model.eval()
    
    # Fetch a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device).float()

    # Get model predictions
    with torch.no_grad():
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()

    # Convert the images and labels back to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    # Display the images along with ground truth and predicted labels
    fig, axes = plt.subplots(num_images, 1, figsize=(10, num_images * 4))
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx].squeeze().numpy()
        label = labels[idx].numpy().astype(int)
        pred = preds[idx].numpy().astype(int)

        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Show ground truth and predictions
        ax.set_title(f"GT: {label}\nPred: {pred}", fontsize=10, loc='left')

    plt.tight_layout()
    plt.show()

# Display a sample of test images with ground truth and predictions
display_sample_images(model, DEVICE, test_loader, num_images=BATCH_SIZE)