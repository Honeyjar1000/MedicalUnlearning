import torch
from torch.utils.data import DataLoader
import resnet_pytorch
from medmnist import RetinaMNIST
import numpy as np
from utils.get_data import get_custom_loaders
from utils.clear_output import clear
from tqdm import tqdm
from unlearn.test_unlearn import get_dataset_f


def GA_Forget_SalUn(model, TrainData:RetinaMNIST, ValData:RetinaMNIST, TestData:RetinaMNIST, criterion, optimizer, mask, device, 
              class_to_forget,
              batch_size,
              epochs): 
    
    TrainData_F = get_dataset_f(TrainData, class_to_forget)
    ValData_F = get_dataset_f(ValData, class_to_forget)

    ForgetTrainLoader, ForgetValLoader, TestLoader = get_custom_loaders(batch_size, TrainData_F, ValData_F, TestData)

    model.train()

    train_acc, train_loss = [], []

    for epoch in range(epochs):

        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for it, (image, label) in tqdm(enumerate(ForgetTrainLoader), total=len(ForgetTrainLoader)):
            counter += 1
            image = image.to(device)

            label = label.to(device).long() # Ensure labels are of type long
            label = label.view(-1)

            output_clean = model(image)
            loss = - criterion(output_clean, label) # Negative loss for GA
        
            optimizer.zero_grad()
            loss.backward()
            
            # SalUn step
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
                    
            optimizer.step()
            output = output_clean.float()
            loss = loss.float()

            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(output_clean.data, 1)
            train_running_correct += (preds == label).sum().item()
        
        # Loss and accuracy for the complete epoch.
        clear()
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(ForgetTrainLoader.dataset))
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}")
        print('-'*50)

    print('FINE TUNING COMPLETE')
    return train_loss, train_acc