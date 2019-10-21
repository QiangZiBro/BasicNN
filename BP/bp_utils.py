import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache
def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ
def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache
def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost
def compute_loss(A2,y):
    loss = np.sum(np.square(A2-y))
    return loss


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def compute_accuracy_rate(y_predict,y):
    assert y_predict.shape[0] == 1
    m = y.shape[1]
    e = 0
    for j in range(m):
        if y_predict[0,j] == y[0,j]:
            e += 1
    return e/m

def initializeData(w,b,N,seed=0):
    """
    args:
        w: a (n,1) shape initialized weight for coefficient of superplace :`wx+b=0`
        b: bias for superplane,also a scalar
        N: number of instances or points.
    return:
        dataset:the last 500 rows is the line seperating the two classes
    """
    np.random.seed(seed=seed)
    numFeatures = len(w)
    w = np.array(w).reshape(1,-1)
    X = np.random.rand(numFeatures,N) * 20  #随机产生N个数据的数据集
    y = np.sign(w.dot(X)+b).reshape(1,-1)    #用标准线 w*x+b=0进行分类

    assert y.shape[0] == 1
    for i,e in enumerate(y[0,:]):
        if e== -1:
            y[0,i] = 0
    return X,y
def generate_train():
    train_X,train_y = initializeData([1,-2.5,2],0,512,seed=0)
    
    return train_X,train_y
def generate_valid():
    valid_X,valid_y = initializeData([1,-2.5,2],0,256,seed=2)
    return valid_X,valid_y
def generate_test():
    test_X,test_y = initializeData([1,-2.5,2],0,256,seed=3)
    return test_X,test_y

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def draw(X,y,ax,title=None):
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    markers = ['o', 'o',]
    colors = ['r', 'g',]
    for i,c in enumerate([0,1]):
        idx = np.where(y[0] == c)
        ax.scatter(X[0, idx], X[1, idx], X[2,idx],marker=markers[i], color=colors[i],  s=25)
    if title:
        plt.title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.savefig("results/"+title+".png")
    plt.show()


def writedata(data,path="./checkpoint"):
    f = open(path, "w")
    f.write(str(data)+"\n")
def tow_dataset():
    def give_sphere(x, y, z, r, num):
        points = []
        for i in range(0, num):
            factor = 1     # A value between 0 and 1 following a gaussian
            ir = r * factor
            itheta = np.arccos(np.random.uniform(-1, 1,))
            iphi = np.random.uniform(0, 2 * np.pi)
            ix = x + ir * np.sin(itheta) * np.cos(iphi)
            iy = y + ir * np.sin(itheta) * np.sin(iphi)
            iz = z + ir * np.cos(itheta)
            points.append((ix, iy, iz))
        return points
    def plot_sphere(points, ax):
        x_list = [x for [x, y, z] in points]
        y_list = [y for [x, y, z] in points]
        z_list = [z for [x, y, z] in points]

        ax.scatter(x_list, y_list, z_list)


    fig = plt.figure()
    ax = Axes3D(fig)


    points1 = give_sphere(0, 0, -2, 2, 1000)
    points2 = give_sphere(0, 0, 2, 2, 1000)
    plot_sphere(points1, ax)
    plot_sphere(points2, ax)

    plt.show()

def compute_local_gradient_output(y,A):

    """
    args:
    A:real output of j
    y:desired output of j
    return:
    delta    local gradient of neural j when j is output layer
    """
    delta2 = - A * (1-A) * (y-A)
    return delta2
def compute_local_gradient_hidden(A,W,delta2):
    delta1 = np.dot(W.T,delta2) * A * (1-A)
    return delta1


def predict(X,  parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Forward propagation
    A1, cache1 = linear_activation_forward(X, W1, b1, "sigmoid")
    A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
    # suppose the output dimension is 1
    for p,e in enumerate(A2[0,:]):
        if A2[0,p] >= 0.5:
            A2[0,p] = 1
        else:
            A2[0,p] = 0
    return A2
def score(A2,y):
    _,m = y.shape
    s = 0.
    for p,e in enumerate(y[0,:]):
        if e == A2[0,p]:
            s+=1
    return s/m
if __name__ == "__main__":
    # X, y = make_classification(n_samples=1000,n_features=3, n_redundant=0, n_informative=2,
    #                                random_state=1, n_clusters_per_class=1)
    # train_X, test_X, train_y, test_y = \
    #     train_test_split(X, y, test_size=.4, random_state=42)
    # train_X = train_X.reshape((3,-1))
    # test_X = test_X.reshape((3,-1))
    # train_y = train_y.reshape((1,-1))
    # test_y = test_y.reshape((1,-1))
    # # test_X,test_y = generate_test()
    # # train_X,train_y = generate_train()
    # print(train_y)
    # draw(train_X,train_y)
    a,b = np.array([1,0,0,1]).reshape(1,-1), np.array([0,0,0,1]).reshape(1,-1)
    print(score(a,b))