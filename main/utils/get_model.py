from resnet_pytorch import ResNet
import torch
import torch.nn as nn

def find_no_class(dataset_name:str):
        if dataset_name.upper() == "RETINAMNIST":
             return 5
        elif dataset_name.upper() == "CHESTMNIST":
            return 14
        elif dataset_name.upper() == "PATHMNIST":
            return 9
        elif dataset_name.upper() == "DERMAMNIST":
            return 7
        elif dataset_name.upper() == "BLOODMNIST":
            return 8
        else:
            raise ValueError(f"Dataset {dataset_name} is not recognized.")


def get_model(model_name, dataset_to_train):
    no_classes = find_no_class(dataset_to_train)
    if model_name.upper() == "RESNET18":
        model = ResNet.from_pretrained('resnet18', num_classes=no_classes)

    if dataset_to_train.upper() == "CHESTMNIST":
        # Modify the first convolution layer to accept grayscale images
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for multi-label classification (14 classes)
        model.fc = nn.Linear(model.fc.in_features, 14)
        
    return model


def load_model(model_id, dataset_to_train):
    no_classes = find_no_class(dataset_to_train)
    model = ResNet.from_pretrained('resnet18', num_classes=no_classes)
    
    if dataset_to_train.upper() == "CHESTMNIST":
        # Modify the first convolution layer to accept grayscale images
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for multi-label classification (14 classes)
        model.fc = nn.Linear(model.fc.in_features, 14)

    model.load_state_dict(torch.load('model_weights/' + model_id, weights_only=True))

    model.eval()
    return model


def get_model_simple():
    model = ResNet.from_pretrained('resnet18', num_classes=5)
    return model