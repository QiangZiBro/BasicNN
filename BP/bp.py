from bp_utils import *
import sys, getopt

def two_layer_model(train_X, train_y, layers_dims,batch_size = 1 ,test_X = None,test_y = None,learning_rate=0.0075, num_iterations=3000, print_cost=False,title="BackPropgation"):

    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    train_scores = []
    test_scores = []

    m = train_X.shape[1]  # number of examples
    batchs = m/batch_size
    (n_x, n_h, n_y) = layers_dims

    # plt.figure(1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw(train_X, train_y, ax, title=title)
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(layers_dims)
    ### END CODE HERE ###
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Loop (gradient descent)

    for i in range(0, num_iterations):
        # batch learning
        for batch in range(int(batchs)):
            X = train_X[:,batch * batch_size : (batch+1)*batch_size]
            Y = train_y[:,batch * batch_size : (batch+1)*batch_size]

            # Forward propagation: LINEAR -> SIGMOID-> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
            A1, cache1 = linear_activation_forward(X, W1, b1, "sigmoid")
            A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

            # Back propagation
            delta2 = compute_local_gradient_output(Y,A2)
            dW2,db2 = 1. / m * np.dot(delta2,A1.T),np.average(delta2,axis=1).reshape(-1,1)
            # print("delta2.shape,A1.T.shape:",delta2.shape,A1.T.shape)
            delta1 = compute_local_gradient_hidden(A1,W2,delta2)
            dW1,db1 = 1. / m * np.dot(delta1,X.T),np.average(delta1,axis=1).reshape(-1,1)
            # print("delta2.shape,A1.T.shape:",delta1.shape,X.T.shape)

            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            # 1.score the model
            pred = predict(train_X, parameters)
            sc = (score(pred,train_y))
            train_scores.append(sc)

            # 2.Compute cost
            A1, cache1 = linear_activation_forward(train_X, W1, b1, "sigmoid")
            A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
            cost = compute_loss(A2, train_y)
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

            # 3.plot some details
            plt.figure(2)
            plt.clf()
            plt.subplot(2,1,1)
            plt.ylabel('cost')
            plt.title("Learning rate =" + str(learning_rate))
            plt.plot(np.squeeze(costs))

            plt.subplot(2,1,2)
            plt.ylabel('accuracy')
            plt.xlabel('iterations (per hundred)')

            plt.plot(np.squeeze(train_scores),'-r',label="train")
            if test_X is not None:
                pred = predict(test_X, parameters)
                sc = (score(pred, test_y))
                test_scores.append(sc)
                plt.plot(np.squeeze(test_scores),'-g',label="test")
                plt.legend()
            plt.pause(0.01)
    plt.savefig("results/" + title +"lr" +str(learning_rate) + "bt" +str( batch_size )+ ".png")
    writedata(parameters,"./results/"+title+".txt")
    scores_cache = (train_scores,test_scores)
    return parameters,costs,scores_cache

def L_layer_model(train_X,train_y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    X,Y = train_X,train_y
    np.random.seed(1)
    costs = []  # keep track of cost
    train_scores = []
    test_scores = []
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        # if print_cost and i % 100 == 0:
        #     # Retrieve W1, b1, W2, b2 from parameters
        #     W1 = parameters["W1"]
        #     b1 = parameters["b1"]
        #     W2 = parameters["W2"]
        #     b2 = parameters["b2"]
        #     # 1.score the model
        #     pred = predict(train_X, parameters)
        #     sc = (score(pred,train_y))
        #     train_scores.append(sc)
        #
        #     # 2.Compute cost
        #     A1, cache1 = linear_activation_forward(train_X, W1, b1, "sigmoid")
        #     A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        #     cost = compute_loss(A2, train_y)
        #     print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        #     costs.append(cost)
        #
        #     # 3.plot some details
        #     plt.figure(2)
        #     plt.clf()
        #     plt.subplot(2,1,1)
        #     plt.ylabel('cost')
        #     plt.title("Learning rate =" + str(learning_rate))
        #     plt.plot(np.squeeze(costs))
        #
        #     plt.subplot(2,1,2)
        #     plt.ylabel('accuracy')
        #     plt.xlabel('iterations (per hundred)')
        #
        #     plt.plot(np.squeeze(train_scores),'-r',label="train")
        #     if test_X is not None:
        #         pred = predict(test_X, parameters)
        #         sc = (score(pred, test_y))
        #         test_scores.append(sc)
        #         plt.plot(np.squeeze(test_scores),'-g',label="test")
        #         plt.legend()
        #     plt.pause(0.01)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def XOR():
    # 1.XOR problem [solved]
    train_X,train_y = np.array([[0,1,0,1],[0,1,1,0],[0,0,0,0]]), np.array([0,0,1,1]).reshape(1,-1)
    test_X,test_y = None,None
    print(train_y)
    title = "XOR"
    parameters = two_layer_model(train_X,train_y,[3,10,1],batch_size=train_X.shape[1],
                                 learning_rate=0.01,num_iterations=100000,print_cost=True,title=title)

def Linear_Separable():
    # 2.linear separable problem [solved]
    test_X,test_y = generate_test()
    train_X,train_y = generate_train()
    title = "linear separable"
    parameters = two_layer_model(train_X,train_y,[3,10,1],batch_size=train_X.shape[1],
                                 learning_rate=0.01,num_iterations=10000,print_cost=True,title=title
                                 )
def NonlinearSeparable():
    # 3.nonlinear separable problem [unsolved]
    X, y = make_classification(n_samples=1000,n_features=3, n_redundant=0, n_informative=2,
                                   random_state=0, n_clusters_per_class=2)
    train_X, test_X, train_y, test_y =train_test_split(X, y, test_size=.4, random_state=42)
    train_X,train_y= train_X.reshape((3,-1)),train_y.reshape((1,-1))
    test_X,test_y = test_X.reshape((3,-1)),test_y.reshape((1,-1))
    title = "nonlinear separable"
    parameters = two_layer_model(train_X,train_y,[3,10,1],
                                 learning_rate=0.01,num_iterations=10000,print_cost=True,
                                 )
def ImageClassfication():
    # 4.image classification
    np.random.seed(1)
    train_X_orig, train_y, test_X_orig, test_y, classes = load_data()
    # Reshape the training and test examples
    train_X_flatten = train_X_orig.reshape(train_X_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_X_flatten = test_X_orig.reshape(test_X_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_X = train_X_flatten / 255.
    test_X = test_X_flatten / 255.

    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    ### CONSTANTS ###
    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    parameters = L_layer_model(train_X, train_y, layers_dims, num_iterations = 2500, print_cost = True)

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "1234", ["xor","linears","nonlinears","imagecls"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-1", "--xor"):
            XOR()
        elif opt in ("-2", "--linears"):
            Linear_Separable()
        elif opt in ("-3", "--nolinears"):
            NonlinearSeparable()
        elif opt in ("-4", "--imagecls"):
            ImageClassfication()
