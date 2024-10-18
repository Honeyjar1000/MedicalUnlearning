

def get_params(model_id):
    model_id_str = model_id.split("_")
    unlearn_batch_size = int(model_id_str[1][2:])
    unlearn_learn_rate = float('0.' + model_id_str[2][2:])
    unlearn_epochs = int(model_id_str[3][6:])
    return unlearn_batch_size, unlearn_learn_rate, unlearn_epochs

