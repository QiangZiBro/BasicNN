from bp_utils import *

#################################################

t = []
train_accuracy_rate = []
valid_accuracy_rate = []
#################################################

# 3 --> 4 -- > 1
def model(X,y,X_valid=None,y_valid=None,batch_size = 1,epochs=1,learning_rate=0.01,show_figure=False):
    print("Input size(dimension,N):",X.shape)
    print("Output size(dimension,N):",y.shape)
    print("batch_size:%d,epochs=%d,learning_rate=%f\n" % (batch_size,epochs,learning_rate) )
    print("Start training...")
    parameters = initialize_parameters([X.shape[0],10,y.shape[0]])

    m = X.shape[1] # instance number
    batch_number = m // batch_size
    for epoch in range(epochs):
        for i in range(batch_number):
            A0 = X[:,i:i+batch_size]
            Y  = y[:,i:i+batch_size]
            W1,b1,W2,b2 = parameters["W1"],parameters["b1"],parameters["W2"],parameters["b2"]
            A1,_ = forward(W1,A0,b1)
            A2,_ = forward(W2,A1,b2)            
            # forward propagate
            delta2 = compute_local_gradient_output(Y,A2)
            delta1 = compute_local_gradient_hidden(A1,W2,delta2)

            # #  average for the batch
            # if batch_size > 1:
            #     delta1 = np.average(delta1,axis=1).reshape(-1,1)
            #     delta2 = np.average(delta2,axis=1).reshape(-1,1)
            #     A0  =   np.average(A0,axis=1).reshape(-1,1)
            #     A1  =   np.average(A1,axis=1).reshape(-1,1)
            #     A2  =   np.average(A2,axis=1).reshape(-1,1)

            cache = (A0,A1,A2)
            delta = (delta1,delta2)          
            # backward propagate
            parameters = update_parameters(parameters,delta,cache,learning_rate=learning_rate)
        
        if (epoch+1)%100 == 0:
            loss = compute_loss(A2,y)
            print("epochs:%d,loss:%f" % (epoch+1,loss))
        if show_figure:
            #reference:https://blog.csdn.net/u013468614/article/details/58689735#11__6
            train__rate = compute_accuracy_rate(predict(parameters,X),y)
            valid_rate = compute_accuracy_rate(predict(parameters,X_valid),y_valid)

            plt.clf() #清空画布上的所有内容
            t_now = epoch
            t.append(t_now)#模拟数据增量流入，保存历史数据
            train_accuracy_rate.append(train__rate)#模拟数据增量流入，保存历史数据
            valid_accuracy_rate.append(valid_rate)
            plt.plot(t,train_accuracy_rate,'-r',label = "train_accuracy")
            plt.plot(t,valid_accuracy_rate,'-g',label = "test_accuracy")
            plt.legend()
            plt.pause(0.01)
    return parameters


def two_layer_model(train_X, train_y, layers_dims,batch_size = 1 ,learning_rate=0.0075, num_iterations=3000, print_cost=False,title="BackPropgation"):

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
    writedata(parameters,title+".txt")
    return parameters

if __name__ == "__main__":
    # 1.XOR problem solved
    # train_X,train_y = np.array([[0,1,0,1],[0,1,1,0],[0,0,0,0]]), np.array([0,0,1,1]).reshape(1,-1)
    # test_X,test_y = None,None
    # print(train_y)
    # title = "XOR"
    ######
    # parameters1 = {'W1': array([[ 1.73243585, -0.68443081, -0.3049401 ],
    #    [-0.50408306,  0.71634201, -1.32879399],
    #    [ 2.35611471, -1.26099638,  0.18419731],
    #    [-0.50976544,  1.57069881, -1.18942279],
    #    [-0.21183062, -0.17974914,  0.65458209],
    #    [-0.64250018, -0.11107727, -0.50683179],
    #    [ 0.04255135,  0.61952975, -0.63544278],
    #    [ 3.13704557,  2.96127197,  0.29011524],
    #    [ 0.64372309, -1.45082764, -0.0709507 ],
    #    [-0.55775774, -0.16705329,  0.30620087]]), 'b1': array([[ 0.54276744],
    #    [ 0.11961244],
    #    [ 0.85749304],
    #    [ 0.22042712],
    #    [-0.03708293],
    #    [ 0.00316321],
    #    [-0.02467956],
    #    [-0.4479583 ],
    #    [-0.46912593],
    #    [ 0.05253329]]), 'W2': array([[-1.10156642, -0.35530683, -1.78710325, -1.23327693, -0.04609117,
    #      0.14979973, -0.61855703,  3.38594615,  1.35710835,  0.39337344]]), 'b2': array([[0.05044877]])}
    # pred = predict(train_X, parameters1)
    # sc = (score(pred, train_y))
    # print(sc)

    # 2.linear separable problem
    # test_X,test_y = generate_test()
    # train_X,train_y = generate_train()
    # title = "linear separable"

    # 3.nonlinear separable problem
    X, y = make_classification(n_samples=1000,n_features=3, n_redundant=0, n_informative=2,
                                   random_state=0, n_clusters_per_class=2)
    train_X, test_X, train_y, test_y =train_test_split(X, y, test_size=.4, random_state=42)
    train_X,train_y= train_X.reshape((3,-1)),train_y.reshape((1,-1))
    test_X,test_y = test_X.reshape((3,-1)),test_y.reshape((1,-1))
    title = "nonlinear separable"


    parameters = two_layer_model(train_X,train_y,[3,10,1],batch_size=train_X.shape[1],
                                 learning_rate=0.01,num_iterations=1000000,print_cost=True,
                                 title=title)


    # debug()