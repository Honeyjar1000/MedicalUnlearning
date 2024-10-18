from tqdm import tqdm
import torch 

def test(model, device, dataloader, criterion, b_multi_label):
    
    print('Test')
    model.eval()  # Set the model to evaluation mode
    correct_prediction = 0
    counter = 0
    test_loss = 0

    with torch.no_grad():  # No need to compute gradients during testing
        for data in tqdm(dataloader, total=len(dataloader)):
            
            # Move data to the same device as the model
            images, labels = data
            images = images.to(device)

            if b_multi_label:
                labels = labels.to(device).float() # Ensure labels are of type float
            elif not b_multi_label:
                labels = labels.to(device).long() # Ensure labels are of type long
                labels = labels.view(-1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()

            # Get predictions
            if b_multi_label:
                # Multi-label classification accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_prediction += (preds == labels).sum().item()
                counter += labels.numel()
            else:
                # Single-label classification accuracy
                _, preds = torch.max(outputs.data, 1)
                correct_prediction += (preds == labels).sum().item()
                counter += len(labels)

    if b_multi_label:
        # Calculate accuracy as the average accuracy across all instances
        average_loss = test_loss / len(dataloader)
        raw_accuracy = 100. * correct_prediction / counter
    else:
        average_loss = test_loss / len(dataloader)
        raw_accuracy = (correct_prediction / counter) * 100

    return raw_accuracy, average_loss