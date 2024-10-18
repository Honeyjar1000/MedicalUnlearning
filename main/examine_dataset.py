import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils.get_data import get_data
import arg_parser
import random
import matplotlib

fig = None
axes = None
key_mapping = {}

def plot_samples(images, labels, class_names, num_samples=16):
    global fig, axes

    grid_size = int(np.sqrt(num_samples))

    if fig is None:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 12))

    for ax in axes.flatten():
        ax.clear()

    rnd_idx_list = [random.randint(0, len(images) - 1) for _ in range(num_samples)]
    idx1 = 0
    

    for i in range(grid_size):
        for j in range(grid_size):
            idx = rnd_idx_list[idx1]
            idx1 += 1
            if idx < len(images):  # Ensure index is within range
                img = images[idx].numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

                # Normalize image to [0, 1] if it is not already
                if img.max() > 1:
                    img = (img - img.min()) / (img.max() - img.min())
                # Ensure the image data is in the range [0, 1]
                img = np.clip(img, 0, 1)

                label = labels[idx]
                # Ensure label_indices are within bounds
                if type(label) != int:
                    # Multiclass class (eg. [3, 4])

                    for m in range(len(label)):
                        label[m] = str(label[m])

                    axes[i, j].imshow(img)
                    axes[i, j].set_title(', '.join(label) if label else "No Label")
                    axes[i, j].axis('off')
                else:
                    # Single class (eg. [3])
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(str(label))
                    axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Pause to allow the figure to be drawn

def display_class_samples(images, labels, class_names, target_class, num_samples=16):
    if type(labels[0]) != int:
        # Multiclass class (eg. [3, 7, 8])
        class_images = [img for img, lbl in zip(images, labels) if target_class in lbl]
        class_labels = [lbl for img, lbl in zip(images, labels) if target_class in lbl]
        plot_samples(class_images, class_labels, class_names, num_samples)
    elif type(labels[0]) == int:
        # Single class (eg. [3])
        class_images = [img for img, lbl in zip(images, labels) if target_class == lbl]
        plot_samples(class_images, [target_class]*len(class_images), class_names, num_samples)

# Wait for key press to show new images
def on_key_press(event, key_bindings, images, labels, class_names):
    if event.key in key_bindings:
        target_class = key_bindings[event.key]
        display_class_samples(images, labels, class_names, target_class)
    elif event.key == ' ':
        plot_samples(images, labels, class_names)


# Key binding logic
def map_keys_to_classes(num_classes):
    key_bindings = {}
        # First map numeric keys (1-9, 0) to classes 0-9
    for i in range(min(10, num_classes)):
        key = str(i + 1) if i < 9 else '0'
        key_bindings[key] = i

    # If there are more than 10 classes, map function keys F1, F2, etc.
    for i in range(10, num_classes):
        key_bindings[f'f{i-9}'] = i
    return key_bindings

def main():
    args = arg_parser.parse_args()

    # Update your get_data function to use this transform
    TrainLoader, ValLoader, TestLoader = get_data(args.dataset, args.batch_size)
    
    train_size = len(TrainLoader.dataset)
    val_size = len(ValLoader.dataset)
    test_size = len(TestLoader.dataset)
    total_size = train_size + val_size + test_size

    print(f"\nTraining set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")
    print(f"Total dataset size: {total_size}")

    # Initialize empty lists for images and labels
    images = []
    labels = []
    class_dict = {}
    for batch in TrainLoader:
        imgs, lbls = batch
        for img, lbl in zip(imgs, lbls):
            labels_list = lbl.tolist()
            if len(labels_list) > 1:
                # Multiclass class (eg. [0, 0, 0, 1, 1, 0])
                active_classes = []
                for i in range(len(labels_list)):
                    if labels_list[i] == 1:
                        active_classes.append(i)
                        if str(i) in class_dict:
                            class_dict[str(i)] += 1
                        else:
                            class_dict[str(i)] = 1
                labels.append(active_classes)
                images.append(img)

            else:
                # Single class (eg. [3])
                labels.append(labels_list[0])
                images.append(img)
                if str(labels_list[0]) in class_dict:
                    class_dict[str(labels_list[0])] += 1
                else:
                    class_dict[str(labels_list[0])] = 1

   # Print the number of images for each class in a formatted manner
    print("\nClass Distribution:")
    for class_id, count in sorted(class_dict.items()):
        print(f"  Class {class_id}: {count} images")

    # Print the classes in a clean format
    class_names = [str(c) for c in sorted(class_dict)]
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes} total):")
    print("  " + ", ".join(class_names))

    if num_classes > 20:
        print("Error: Too many classes to map key bindings.")
        return

    key_bindings = map_keys_to_classes(num_classes)
    print("\nKey Bindings:")
    for key, class_id in sorted(key_bindings.items()):
        print(f"  Key '{key}': Class {class_id}")

    # Initial random display
    plot_samples(images, labels, class_names)
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, key_bindings, images, labels, class_names))
    plt.show()

if __name__ == "__main__":
    main()