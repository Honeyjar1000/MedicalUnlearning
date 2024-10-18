import torch
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np
from utils.get_data import get_custom_loaders, get_data
from utils.get_model import load_model
import arg_parser
import torch.nn as nn
from utils.get_criterion import get_criterion

def save_gradient_ratio(model, device, criterion, optimizer, learning_rate, save_dir_str, DataLoader, b_multi_label):
    
    gradients = {}
    model.to(device)
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param)

    for i, (image, label) in enumerate(DataLoader):

        image = image.to(device)
        # Determine whether it's a multi-label or single-label case
        if b_multi_label:
            label = label.to(device).float()  # Ensure labels are of type float
        else:
            label = label.to(device).long()  # Ensure labels are of type long
            label = label.view(-1)
        
        # compute output
        output_clean = model(image)

        # Ensure the label tensor is on the same device as the model output ?>????????????????? 
        if not b_multi_label:
            label = label. view(-1)
        
        # Move label to the same device as the model output
        label = label.to(device)

        # Compute loss
        loss = - criterion(output_clean, label)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        print("Threshold: ", i)
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top percentage elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        os.makedirs(save_dir_str, exist_ok=True)
        torch.save(hard_dict, os.path.join(save_dir_str, "threshold_{}.pt".format(i)))


def main():

    global args
    args = arg_parser.parse_args()
    model = load_model(args.model_id, args.dataset)
    
    save_dir_str = "SaliencyMaps\\" + str(args.model_id)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TrainLoader, ValLoader, TestLoader = get_data(args.dataset, args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion, b_multi_label = get_criterion(args.dataset)
    save_gradient_ratio(model, device, criterion, optimizer, args.lr, save_dir_str, TrainLoader, b_multi_label)


if __name__ == "__main__":
    main()

