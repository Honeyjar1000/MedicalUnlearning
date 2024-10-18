import os
import time
import torch
import arg_parser
from trainer.train import train
from trainer.val import validate
from trainer.test import test
from utils.get_model import get_model
from utils.get_data import get_data
from utils.get_model_id import get_id
from utils.save_plot import SavePlots
from utils.clear_output import clear
from utils.setup import setup_seed
from utils.get_criterion import get_criterion

def check_model_id_exists(model_id):
    """Check if a model with the given ID already exists."""
    return os.path.exists(f'model_weights/{model_id}')

def prompt_user_for_model_id(model_id):
    """Prompt the user to choose an action if the model ID already exists."""
    print(f"Model ID '{model_id}' already exists.")
    choice = input("Do you want to (O)verwrite, (C)hoose a new ID, or (Q)uit? [O/C/Q]: ").strip().lower()
    
    if choice == 'o':
        os.remove(f'model_weights/{model_id}')
        print(f"Old model '{model_id}' has been removed.")
        return model_id
    elif choice == 'c':
        new_id = model_id
        i = 1
        while check_model_id_exists(new_id):
            new_id = f"{model_id}_{i}"
            i += 1
        print(f"New model ID '{new_id}' will be used.")
        return new_id
    elif choice == 'q':
        print("Exiting the program.")
        exit()
    else:
        print("Invalid choice. Exiting.")
        exit()

def main():
    global args
    args = arg_parser.parse_args()
    
    setup_seed(args.seed)
        
    start_time = time.time()

    # Get initial model ID
    if args.save_model_id == "None":
        model_id = get_id(args)
    else:
        model_id = args.save_model_id
    
    # Check if the model ID already exists
    if check_model_id_exists(model_id):
        model_id = prompt_user_for_model_id(model_id)
    
    TrainLoader, ValLoader, TestLoader = get_data(args.dataset, args.batch_size)
    
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("\nBegin Train")
    print("Cuda is available: ", torch.cuda.is_available())
    print("Using device: ", str(device))
    model = get_model(args.arch, args.dataset).to(device)
    criterion, b_multi_label = get_criterion(args.dataset)

    best_valid_acc = 0.0  # Initialize a variable to keep track of the best validation accuracy

    # Training Loop
    for epoch in range(args.epochs):
        print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")

        train_epoch_loss, train_epoch_acc = train(
            args, 
            model,
            device,
            criterion,
            TrainLoader,
            b_multi_label
        )

        valid_epoch_loss, valid_epoch_acc = validate(
            args, 
            model,
            device,
            criterion,
            ValLoader,
            b_multi_label
        )
        

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        # Save the model if this epoch's validation accuracy is the best we've seen so far
        if valid_epoch_acc > best_valid_acc:
            best_valid_acc = valid_epoch_acc
            torch.save(model.state_dict(), f'model_weights/{model_id}_best')
            print(f"New best model saved with validation accuracy: {best_valid_acc:.3f}")

    test_epoch_acc, test_epoch_loss = test(
            model,
            device,
            TestLoader,
            criterion,
            b_multi_label
        )
    print(f"Test loss: {test_epoch_loss:.3f}, Test acc: {test_epoch_acc:.3f}")

    print('TRAINING COMPLETE')

    SavePlots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=model_id
    )

    # Save the last model as well (optional)
    torch.save(model.state_dict(), f'model_weights/{model_id}_last')
    print(f"Last model saved in model_weights/{model_id}_last")
    
    print("Training time: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()