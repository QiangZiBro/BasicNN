import numpy as np
import matplotlib.pyplot as plt
from multi_layer_nn import L_layer_model
from sklearn.model_selection import train_test_split

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def explore():
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        single_img = X[i]
        single_img = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
        fig.add_subplot(rows, columns, i)
        plt.imshow(single_img)
    plt.show()


if __name__ == "__main__":
    data = unpickle("datasets\cifar-10-python\cifar-10-batches-py\data_batch_1")
    X = data[b"data"]
    y = np.array(data[b"labels"])
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.4, random_state=42) #train_X.shape,train_y.shape (6000, 3072) (6000,)
    layers_dims = [3072, 20, 7, 5, 10]  # 5-layer model
    parameters = L_layer_model(train_X.T, train_y, layers_dims, num_iterations = 2500, print_cost = True)