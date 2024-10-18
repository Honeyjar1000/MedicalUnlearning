import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.get_model import load_model
from utils.get_data import get_data, get_test_data
import random
import string
from utils.get_criterion import get_criterion
import arg_parser
from utils.setup import setup_seed

fig = None
axes = None
key_mapping = {}

def evaluate_model(model, dataloader, device, criterion, b_multi_label):
    """
    Evaluate the model on the test dataset and collect images, predictions, and ground truths.

    Parameters:
    - model: The trained model.
    - dataloader: DataLoader for the test set.
    - device: Device to run the computation on (CPU or GPU).

    Returns:
    - accuracy: Accuracy of the model on the test dataset.
    - auc: AUC score of the model on the test dataset (for binary classification).
    - images: List of all images.
    - labels: List of all ground truth labels.
    - preds: List of all predicted labels.
    - class_accuracies: Accuracy for each class.
    - class_counts: Number of samples for each class.
    """
    model.eval()  # Set the model to evaluation mode
    correct_prediction = 0
    total_samples = 0
    all_labels = []
    all_outputs = []
    all_images = []  # Store all images
    all_image_labels = []  # Store all image labels
    all_image_preds = []  # Store all image predictions

    test_loss = 0
    
    # Determine unique classes in the dataset
    unique_classes = set()
    for i in range(len(dataloader.dataset)):
        unique_classes.add(dataloader.dataset[i][1][0])
    
    num_classes = len(unique_classes)
    
    # Initialize counters for class-wise accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():  # No need to compute gradients during testing
        for imgs, labels in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):

            imgs = imgs.to(device)
            
            if b_multi_label:
                labels = labels.to(device).float() # Ensure labels are of type float
            elif not b_multi_label:
                labels = labels.to(device).long() # Ensure labels are of type long
                labels = labels.view(-1)

                
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            
            # Get predictions
            if b_multi_label:
                # Multi-label classification accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid and threshold
                correct_preds = (preds == labels).float().sum().item()  # Count correct predictions
                correct_prediction += correct_preds
            else:
                # Single-label classification accuracy
                _, preds = torch.max(outputs.data, 1)
                correct_prediction += (preds == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(preds.cpu().numpy())
            
            # Collect all images, labels, and predictions
            all_images.extend(imgs.cpu())
            all_image_labels.extend(labels.cpu().numpy())
            all_image_preds.extend(preds.cpu().numpy())
            total_samples += len(labels)

            if b_multi_label:
                pass
            else:
                # Update class-wise accuracy counters
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

    accuracy = (correct_prediction / total_samples) * 100
    
    # Compute AUC - how to do it for non multi label?
    if b_multi_label:  # Multiclass classification check
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
    else:
        auc = None

    # Compute class-wise accuracy
    class_accuracies = class_correct / class_total * 100
    
    # Handle possible division by zero for classes with no samples
    class_accuracies[np.isnan(class_accuracies)] = 0

    return accuracy, auc, all_images, all_image_labels, all_image_preds, class_accuracies, class_total

def display_image_grid(images, labels, preds, args, num_cols=5):
    """
    Display or update a grid of images with predictions and ground truth labels.

    Parameters:
    - images: List of images to display.
    - labels: List of ground truth labels.
    - preds: List of predicted labels.
    - num_cols: Number of columns in the grid.
    """
    global fig, axes
    
    num_images = min(len(images), 20)  # Limit to 20 images
    num_rows = 4  # Fixed to 4 rows

    if fig is None:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    
    for ax in axes.flatten():
        ax.clear()

    for i, (img, label, pred) in enumerate(zip(images[:20], labels[:20], preds[:20])):
        ax = axes.flatten()[i]
        img = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC and to numpy array
        
        # Normalize image to [0, 1] if it is not already
        if img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        # Ensure the image data is in the range [0, 1]
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        

        # Color code text based on correctness
        b_multi_label = False
        if (args.dataset.upper() == "CHESTMNIST"):
            b_multi_label = True

        if b_multi_label:
            color = 'green'
            for i in range(len(label)):
                if label[i] != pred[i]:
                    color = 'red'
        else:
            if label == pred:
                color = 'green'  
            else:
                color = 'red'

        ax.set_title(f"GT: {label}\nPred: {pred}", color=color)
        ax.axis('off')

    # Hide unused subplots
    for j in range(num_images, num_rows * num_cols):
        axes.flatten()[j].axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Pause to allow the figure to be drawn
    
def on_key(event, all_images, all_labels, all_preds, args, class_counts, class_accuracies):
    """
    Handle key press events to display new samples.

    Parameters:
    - event: The key press event.
    - all_images: List of all images.
    - all_labels: List of ground truth labels.
    - all_preds: List of predicted labels.
    - class_counts: Number of samples for each class.
    - class_accuracies: Accuracy for each class.
    """
    if event.key == ' ':  # Spacebar
        # Randomly sample a new set of images
        sampled_indices = random.sample(range(len(all_images)), min(20, len(all_images)))
        new_images = [all_images[i] for i in sampled_indices]
        new_labels = [all_labels[i] for i in sampled_indices]
        new_preds = [all_preds[i] for i in sampled_indices]
        display_image_grid(new_images, new_labels, new_preds, args)
        
    elif event.key in key_mapping:
        class_id = key_mapping[event.key]
        
        # Create a list of (image, label, pred) tuples
        indexed_samples = list(zip(all_images, all_labels, all_preds))
        
        # Filter based on class_id
        filtered_samples = [(img, label, pred) for img, label, pred in indexed_samples if label == class_id]

        # Ensure we have up to 20 images
        if len(filtered_samples) > 20:
            filtered_samples = random.sample(filtered_samples, 20)

        if filtered_samples:
            filtered_images, filtered_labels, filtered_preds = zip(*filtered_samples)
            display_image_grid(filtered_images, filtered_labels, filtered_preds, args)
        else:
            print(f"Class ID {class_id} does not exist.")

def setup_key_mapping(num_classes, class_counts, class_accuracies):
    """
    Set up the keyboard key mapping based on the number of classes.

    Parameters:
    - num_classes: Number of classes in the dataset.
    - class_counts: Number of samples for each class.
    - class_accuracies: Accuracy for each class.
    """
    global key_mapping
    key_mapping = {}
    
    if num_classes > 20:
        print("Error: Too many classes for keyboard mapping.")
        return
    
    # Map 1-9, 0 to the first 10 classes
    for i in range(min(num_classes, 10)):
        key_mapping[str(i + 1)] = i
    
    # Map F1-F10 to classes 11-20
    for i in range(max(num_classes - 10, 0)):
        key_mapping[f'f{i + 1}'] = i + 10

    # Print key mapping and class statistics
    print("Key mapping for classes:")
    for key, class_id in key_mapping.items():
        print(f"  {key}: Class {class_id}")

    print("Class statistics:")
    for i, (count, accuracy) in enumerate(zip(class_counts, class_accuracies)):
        print(f"  Class {i}: {count} samples, Accuracy: {accuracy:.2f}%")

def main():

    args = arg_parser.parse_args()
    setup_seed(args.seed)
    test_loader = get_test_data(args.dataset, args.batch_size, num_workers=args.num_workers)
    print("Dataset test size: ", len(test_loader.dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_id, args.dataset)
    model.to(device)
    criterion, b_multi_label = get_criterion(args.dataset)
    
    # Evaluate the model
    accuracy, auc, all_images, all_labels, all_preds, class_accuracies, class_counts = evaluate_model(model, test_loader, device, criterion, b_multi_label)

    # Print overall accuracy and AUC
    print(f"Model Accuracy: {accuracy:.2f}%")
    if auc is not None:
        print(f"Model AUC: {auc:.2f}")
    
    # Set up key mapping and start interactive mode
    num_classes = len(class_counts)
    setup_key_mapping(num_classes, class_counts, class_accuracies)
    
    # Display initial grid
    display_image_grid(all_images[:20], all_labels[:20], all_preds[:20], args)

    # Connect the key press event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: on_key(event, all_images, all_labels, all_preds, args, class_counts, class_accuracies))
    
    plt.show()


if __name__ == "__main__":

    main()