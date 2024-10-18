import torch
from torch.utils.data import DataLoader
import resnet_pytorch
from medmnist import RetinaMNIST
import numpy as np
from utils.get_data import get_custom_loaders
from tqdm import tqdm
from utils.clear_output import clear

def ForgetClass(dataset: RetinaMNIST, 
                class_to_forget: int):
    
    n = dataset.imgs.shape[0]
    index_to_delete = []

    for i in range(n):
        label = dataset.labels[i]
        if label[0] == class_to_forget:
            index_to_delete.append(i)


    dataset.imgs = np.delete(dataset.imgs, index_to_delete, 0)
    dataset.labels = np.delete(dataset.labels, index_to_delete, 0)

    dataset.info["n_samples"][dataset.split] = dataset.imgs.shape[0]
    
    return dataset

def RandomiseLabels(dataset:RetinaMNIST, label_to_change):
    n = dataset.imgs.shape[0]
    index_to_change = []

    for i in range(n):
        label = dataset.labels[i]
        if label[0] == label_to_change:
            index_to_change.append(i)


    for index in index_to_change:
        dataset.labels[index][0] = np.random.randint(0,4)
    
    return dataset

def RL_Forget(model, TrainData:RetinaMNIST, ValData:RetinaMNIST, TestData:RetinaMNIST, criterion, optimizer, mask, device, 
              class_to_forget,
              batch_size,
              epochs): 
    
    TrainData_RL = RandomiseLabels(TrainData, class_to_forget)
    ValData_RL = RandomiseLabels(ValData, class_to_forget)

    ForgetTrainLoader, ForgetValLoader, ForgetTestLoader = get_custom_loaders(batch_size, TrainData_RL, ValData_RL, TestData)

    model.train()

    train_acc, train_loss = [], []

    for epoch in range(epochs):

        print("Epoch ", epoch," / ", epochs)
        
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0

        for it, (image, label) in tqdm(enumerate(ForgetTrainLoader), total=len(ForgetTrainLoader)):
            counter += 1
            image = image.to(device)
            
            label = label.to(device).long() # Ensure labels are of type long
            label = label.view(-1)


            output_clean = model(image)


            loss = criterion(output_clean, label)
        
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

        clear()
        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(ForgetTrainLoader.dataset))
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}")
        print('-'*50)

    print('FINE TUNING COMPLETE')
    return train_loss, train_acc