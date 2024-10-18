import torch.nn as nn

def get_criterion(dataset_name):
    
    if dataset_name.upper() == "RETINAMNIST":
        return nn.CrossEntropyLoss(), False
    elif dataset_name.upper() == "CHESTMNIST":
        return nn.BCEWithLogitsLoss(), True
    elif dataset_name.upper() == "PATHMNIST":
        return nn.CrossEntropyLoss(), False
    elif dataset_name.upper() == "DERMAMNIST":
        return nn.CrossEntropyLoss(), False
    elif dataset_name.upper() == "BLOODMNIST":
        return nn.CrossEntropyLoss(), False