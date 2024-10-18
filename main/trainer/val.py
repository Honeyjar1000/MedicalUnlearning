from tqdm import tqdm
import torch
import numpy as np
from IPython.display import clear_output
import torch.optim as optim
import torch.nn as nn

def validate(args, model, device, criterion, dataloader, b_multi_label):
    model.eval()
    print('Validation')

    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    total_samples = 0  # To track the total number of samples

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            
            images, labels = data
            images = images.to(device)

            if b_multi_label:
                labels = labels.to(device).float() # Ensure labels are of type float
            elif not b_multi_label:
                labels = labels.to(device).long() # Ensure labels are of type long
                labels = labels.view(-1)
                
            # Forward pass.
            outputs = model(images)
            
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.

            if b_multi_label:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                valid_running_correct += (preds == labels).sum().item()
                total_samples += labels.numel()
            else:
                # Single-label classification accuracy
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    
    # Loss and accuracy for the complete epoch.
    if b_multi_label:
        # Calculate accuracy as the average accuracy across all instances
        epoch_loss = valid_running_loss / len(dataloader)
        epoch_acc = 100. * valid_running_correct / total_samples
    else:
        # Standard accuracy calculation for single-label case
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(dataloader.dataset))
    
    return epoch_loss, epoch_acc