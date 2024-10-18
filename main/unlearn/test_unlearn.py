from tqdm import tqdm
import numpy as np

def Test_acc(model, device, dataloader):
    counter = 0
    correct_prediction = 0

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        
            
        imgs, label = data

        imgs = imgs.to(device)
        label = label.to(device).long() # Ensure labels are of type long
        label = label.view(-1)

        output = model(imgs)

        for i in range(len(output)):
            output_array = []
            for x in output[i]:
                output_array.append(round(float(x), 5))
            
            y_hat =  output_array.index(max(output_array))
            actual = int(label[i])
            #print(i, " | prediction: ", y_hat, " | Actual: ", actual, " probability output: ", output_array)
            if y_hat == actual:
                correct_prediction += 1
            counter += 1
            

    accuracy = correct_prediction / counter
    return accuracy * 100



def get_dataset_f(dataset, class_to_forget):
    n = dataset.imgs.shape[0]
    index_to_delete = []

    for i in range(n):
        label = dataset.labels[i]
        if label[0] != class_to_forget:
            index_to_delete.append(i)

    dataset.imgs = np.delete(dataset.imgs, index_to_delete, 0)
    dataset.labels = np.delete(dataset.labels, index_to_delete, 0)

    dataset.info["n_samples"][dataset.split] = dataset.imgs.shape[0]
    return dataset