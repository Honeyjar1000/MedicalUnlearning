from torchvision import transforms
from medmnist import RetinaMNIST, ChestMNIST, PathMNIST, DermaMNIST, BloodMNIST
from torch.utils.data import DataLoader
import torch

data_transform_retina = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transform_chest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom transformation to handle both single-channel and 3-channel images
class ChestTransform:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel mean/std for grayscale
        ])

    def __call__(self, img):
        img = self.transform(img)
        if img.shape[0] == 1:  # Single-channel image
            img = img.repeat(3, 1, 1)  # Repeat the channel to convert it to 3-channel
        # Clip the values to the range [0, 1]
        img = torch.clamp(img, 0, 1)
        return img

def get_custom_loader(batch_size, dataset, num_workers=0):
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return data_loader

def get_custom_loaders(batch_size, trainset, valset, testset, num_workers=0):
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader

def get_data(dataset_name: str, batchsize: int, num_workers: int = 0):
    """
    Get data loaders for the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to load.
    - batchsize: Batch size for the DataLoader.
    - num_workers: Number of workers for DataLoader.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    """
    if dataset_name.upper() == "RETINAMNIST":
        trainset = RetinaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = RetinaMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = RetinaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "CHESTMNIST":
        trainset = ChestMNIST(split="train", transform=data_transform_chest, size=224, download=True)
        valset = ChestMNIST(split="val", transform=data_transform_chest, size=224, download=True)
        testset = ChestMNIST(split="test", transform=data_transform_chest, size=224, download=True)
    elif dataset_name.upper() == "PATHMNIST":
        trainset = PathMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = PathMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = PathMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "DERMAMNIST":
        trainset = DermaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = DermaMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = DermaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "BLOODMNIST":
        trainset = BloodMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = BloodMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = BloodMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")

    return get_custom_loaders(batchsize, trainset, valset, testset, num_workers)

def get_test_data(dataset_name: str, batchsize: int, num_workers: int = 0):
    """
    Get data loaders for the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to load.
    - batchsize: Batch size for the DataLoader.
    - num_workers: Number of workers for DataLoader.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    """
    if dataset_name.upper() == "RETINAMNIST":
        testset = RetinaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "CHESTMNIST":
        testset = ChestMNIST(split="test", transform=data_transform_chest, size=224, download=True)
    elif dataset_name.upper() == "PATHMNIST":
        testset = PathMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "DERMAMNIST":
        testset = DermaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "BLOODMNIST":
        testset = BloodMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")

    return get_custom_loader(batchsize, testset, num_workers)

def get_dataset(dataset_name: str):
    
    if dataset_name.upper() == "RETINAMNIST":
        trainset = RetinaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = RetinaMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = RetinaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "CHESTMNIST":
        trainset = ChestMNIST(split="train", transform=data_transform_chest, size=224, download=True)
        valset = ChestMNIST(split="val", transform=data_transform_chest, size=224, download=True)
        testset = ChestMNIST(split="test", transform=data_transform_chest, size=224, download=True)
    elif dataset_name.upper() == "PATHMNIST":
        trainset = PathMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = PathMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = PathMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "DERMAMNIST":
        trainset = DermaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = DermaMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = DermaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "BLOODMNIST":
        trainset = BloodMNIST(split="train", transform=data_transform_retina, size=224, download=True)
        valset = BloodMNIST(split="val", transform=data_transform_retina, size=224, download=True)
        testset = BloodMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")

    return trainset, valset, testset


def get_train_dataset(dataset_name: str):
    
    if dataset_name.upper() == "RETINAMNIST":
        trainset = RetinaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "CHESTMNIST":
        trainset = ChestMNIST(split="train", transform=data_transform_chest, size=224, download=True)
    elif dataset_name.upper() == "PATHMNIST":
        trainset = PathMNIST(split="train", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "DERMAMNIST":
        trainset = DermaMNIST(split="train", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "BLOODMNIST":
        trainset = BloodMNIST(split="train", transform=data_transform_retina, size=224, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")

    return trainset


def get_test_dataset(dataset_name: str):
    
    if dataset_name.upper() == "RETINAMNIST":
        trainset = RetinaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "CHESTMNIST":
        trainset = ChestMNIST(split="test", transform=data_transform_chest, size=224, download=True)
    elif dataset_name.upper() == "PATHMNIST":
        trainset = PathMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "DERMAMNIST":
        trainset = DermaMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    elif dataset_name.upper() == "BLOODMNIST":
        trainset = BloodMNIST(split="test", transform=data_transform_retina, size=224, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")

    return trainset