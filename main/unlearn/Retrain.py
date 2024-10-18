from medmnist import RetinaMNIST
import numpy as np
from utils.get_data import get_custom_loaders
from trainer.train import train
from trainer.val import validate
from trainer.test import test
from utils.clear_output import clear
import arg_parser
from utils.get_criterion import get_criterion

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

def Retrain(args, model, TrainData, ValData, TestData, 
            device,  class_to_forget, batch_size, epochs):

    TrainDataForgot = ForgetClass(TrainData, class_to_forget)
    ValDataForgot = ForgetClass(ValData, class_to_forget)
    TestDataForgot = ForgetClass(TestData, class_to_forget)

    TrainLoaderForgot, ValLoaderForgot, TestLoaderForgot = get_custom_loaders(batch_size, TrainDataForgot, ValDataForgot, TestDataForgot)
    criterion, b_multi_label = get_criterion(args.dataset)

    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []
    test_acc = [] 

    # Training Loop
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            args,
            model, 
            device,
            criterion,
            TrainLoaderForgot, 
            b_multi_label
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            args,
            model, 
            device,
            criterion,
            ValLoaderForgot, 
            b_multi_label
        )
        
        test_epoch_acc, test_epoch_loss = test(
            model,
            device,
            TestLoaderForgot,
            criterion,
            b_multi_label
        )

        clear()
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        test_acc.append(test_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print(f"Test acc: {test_epoch_acc:.3f}")
        print('-'*50)
        print('TRAINING COMPLETE')

    print('RETRAIN FINISHED')
    return train_loss, train_acc, valid_loss, valid_acc, test_acc