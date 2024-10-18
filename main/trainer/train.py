from tqdm import tqdm
import torch
import numpy as np
from IPython.display import clear_output
import torch.optim as optim
import torch.nn as nn

def train(args, model, device, criterion, trainloader, b_multi_label):
    print('Train')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total_samples = 0  # To track the total number of samples

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        images, labels = data

        images = images.to(device)

        # Determine whether it's a multi-label or single-label case
        if b_multi_label:
            labels = labels.to(device).float() # Ensure labels are of type float
        elif not b_multi_label:
            labels = labels.to(device).long() # Ensure labels are of type long
            labels = labels.view(-1)

        optimizer.zero_grad()           # Clear old gradients
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        train_running_loss += loss.item()

        if b_multi_label:
            # Multi-label classification accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_running_correct += (preds == labels).sum().item()
            total_samples += labels.numel()
        else:
            # Single-label classification accuracy
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()

        loss.backward()                 # Backward pass: compute gradients
        optimizer.step()                # Update weights
    
    # Loss and accuracy for the complete epoch.
    if b_multi_label:
        # Calculate accuracy as the average accuracy across all instances
        epoch_loss = train_running_loss / len(trainloader)
        epoch_acc = 100. * train_running_correct / total_samples
    else:
        # Standard accuracy calculation for single-label case
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc