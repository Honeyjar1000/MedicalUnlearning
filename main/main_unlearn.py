import arg_parser
from utils.get_data import get_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from utils.get_model import load_model, get_model
from unlearn.RL_with_SalUn import RL_Forget as RL_Forget_SalUn
from unlearn.RL import RL_Forget
from unlearn.FT import FT_Forget
from unlearn.FT_with_SalUn import FT_Forget_SalUn
from unlearn.Retrain import Retrain
from unlearn.GA_with_SalUn import GA_Forget_SalUn
from unlearn.GA import GA_Forget
from utils.get_train_params_from_id import get_params
from utils.get_criterion import get_criterion
import time
from utils.setup import setup_seed

def main():
    global args
    args = arg_parser.parse_args()
    setup_seed(args.seed)
    thresh_str = str(str(args.mask_thresh).split('.')[-1])
    start_time = time.time()
    mask_path = 'SaliencyMaps/'+str(args.model_id)+'/threshold_0.'+thresh_str+'.pt'

    criterion, b_multi_label = get_criterion(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_id, args.dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)
    mask = torch.load(mask_path, weights_only=True)
    train_data, val_data, test_data = get_dataset(args.dataset)


    if args.unlearn.upper() == "RL_WITH_SALUN":
        RL_Forget_SalUn(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.batch_size,
                    epochs=args.unlearn_epochs) 
            
    elif args.unlearn.upper() == "RL":
        RL_Forget(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.unlearn_batch_size,
                    epochs=args.unlearn_epochs) 
    
    elif args.unlearn.upper() == "FT_WITH_SALUN":
        FT_Forget_SalUn(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.unlearn_batch_size,
                    epochs=args.unlearn_epochs) 
    
    elif args.unlearn.upper() == "FT":
        FT_Forget(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.unlearn_batch_size,
                    epochs=args.unlearn_epochs) 
    
    elif args.unlearn.upper() == "GA_WITH_SALUN":
        GA_Forget_SalUn(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.unlearn_batch_size,
                    epochs=args.unlearn_epochs) 
    
    elif args.unlearn.upper() == "GA":
        GA_Forget(  model=model,
                    TrainData=train_data,
                    ValData=val_data,
                    TestData=test_data,
                    criterion=criterion,
                    optimizer = optimizer,
                    mask = mask,
                    device=device,
                    class_to_forget=args.class_to_forget,
                    batch_size=args.unlearn_batch_size,
                    epochs=args.unlearn_epochs) 
        
    elif args.unlearn.upper() == "RETRAIN":
        model = get_model(args.arch, args.dataset).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.unlearn_lr)

        train_loss, train_acc, valid_loss, valid_acc, test_acc = Retrain(
            args=args,    
            model=model,
            TrainData=train_data,
            ValData=val_data,
            TestData=test_data,
            device=device,
            class_to_forget=args.class_to_forget,
            batch_size=args.unlearn_batch_size,
            epochs=args.unlearn_epochs
        )
        
    unlearn_ep = str(args.unlearn_epochs)
    unlearn_lr = str(args.unlearn_lr).split(".")[-1]
    model_name=args.model_id+"_FORGOTTEN_"+str(args.class_to_forget)+"_"+args.unlearn.upper()+"_mask"+thresh_str+"_unlearnEpochs"+unlearn_ep+"_unlearnlr"+unlearn_lr
    torch.save(model.state_dict(), 'model_weights/'+model_name)
    print("Unlearned model saved as: ", model_name)
    print("Unlearn time: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()

