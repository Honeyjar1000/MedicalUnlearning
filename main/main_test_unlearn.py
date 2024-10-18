import arg_parser
from unlearn import test_unlearn
from unlearn.RL_with_SalUn import ForgetClass
from utils.get_model import load_model
from utils.get_data import get_data, get_custom_loader, get_dataset, get_test_dataset, get_train_dataset
import torch
from utils.setup import setup_seed

def main():
    global args
    args = arg_parser.parse_args()
    setup_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_id, args.dataset).to(device)

    train_data = get_train_dataset(args.dataset)
    dataset_f = test_unlearn.get_dataset_f(train_data, args.class_to_forget)
    train_loader_forgotten_class= get_custom_loader(args.batch_size, dataset_f)
    unlean_acc = test_unlearn.Test_acc(model, device, train_loader_forgotten_class)


    train_data = get_train_dataset(args.dataset)
    test_data = get_test_dataset(args.dataset)
    train_data_forgotten = ForgetClass(train_data, args.class_to_forget)
    test_data_forgotten = ForgetClass(test_data, args.class_to_forget)
    train_loader_forgotten = get_custom_loader(args.batch_size, train_data_forgotten)
    test_loader_forgotten = get_custom_loader(args.batch_size, test_data_forgotten)
    remaining_acc = test_unlearn.Test_acc(model, device, train_loader_forgotten)
    test_acc = test_unlearn.Test_acc(model, device, test_loader_forgotten)

    print("Unlearn Accuracy:\t{}%".format(round(100-unlean_acc, 3)))
    print("Remaining Accuracy:\t{}%".format(round(remaining_acc, 3)))
    print("Test Accuracy:\t\t{}%".format(round(test_acc, 3)))

if __name__ == "__main__":
    main()

 