

def get_id(args):
    
    aug_str = ''
    if args.data_aug:
        aug_str = 'augT'
    else:
        aug_str = 'augF'
    lr_str = str(args.lr).split(".")[1]
    model_id = 'Retina_bs'+str(args.batch_size)+'_lr'+str(lr_str)+'_epochs'+str(args.epochs)+'_' + str(aug_str)

    print("Model ID: ", model_id)
    return model_id