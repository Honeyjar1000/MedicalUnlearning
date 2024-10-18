import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of SalUn Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument(
        "--dataset", type=str, default="RetinaMNIST", help="dataset"
    )
    parser.add_argument(
        "--input_size", type=int, default=224, help="size of input images"
    )
    parser.add_argument(
        "--num_classes", type=int, default=5
    )
    parser.add_argument(
        "--num_workers", default=0, type=int, help="Number of workers"
    )

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )

    ##################################### General setting ############################################
    parser.add_argument(
        "--seed", default=2, type=int, help="random seed"
    )
    parser.add_argument(
        "--train_seed", default=1, type=int, help="seed for training (default value same as args.seed)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device id"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint file"
    )
    parser.add_argument(
        "--model_id", help="The name of the model", default=None, type=str,
    )

    ##################################### Training setting #################################################
    parser.add_argument(
        "--batch_size", type=int, default=50, help="batch size"
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--data_aug", default=False, type=bool, help="Augment dataset to increase size"
    )
    parser.add_argument(
        "--save_model_id", default="None", type=str, help="model id to save trained model as"
    )
    


    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="RL_with_SalUn", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs", default=10, type=int, help="number of total epochs for unlearn to run",
    )
    parser.add_argument(
        "--unlearn_batch_size", default=100, type=int, help="batch size for data when running unlearn",
    )
    parser.add_argument(
        "--class_to_forget", type=int, default=-1, help="Specific class to forget"
    )
    parser.add_argument(
        "--mask_thresh", type=float, default=0.5, help="Which threshhold to use for the mask"
    )

    return parser.parse_args()
