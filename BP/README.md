# Back Propagation in neural network
## 0 How To Run

## 1 My Work
### 1.1 Space point classification(binary classification)
$\quad$Given a dataset randomly generated, which contains two types of points, labeled by 1 or labeled by 0. The task is to classify them. I design a one-hidden-layer neural network. In this network, the input dimension is 3, the next is 4, and the last is 2. In another word, one point that has three dimensions is inputted in the network, then it is mapped into four dimension space. Intuitionally, it's easier to classify them. And the output layer size is two, which outputs 01(label 1) or 10(label 0).

### Implementation details
$\quad$There are some basic functions I have implemented for the back propagation algorithm.They are as follow:

| function | description | args | return |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `sigmoid(Z)`                                                   | sigmoid function                                                                    | Z:a scalar, vector, or matrix                                                                                                    |                                                          |
| `forward(W,X,b)`                                               | forward for the **single**  layer                                                   | W:   the coefficient matrix of the network   X:   input point with three dimensions:x1,x2,x3  b:   bias                          | Z,A                                                      |
| `compute_local_gradient_output(y,A)`                           | compute $\delta$ with respect to output layer                                       | A: real output of j y: desired output of j                                                                                       | delta: local gradient of neural j when j is output layer |
| `compute_local_gradient_hidden(A,W,delta2)`                    | compute $\delta$ with respect to hidden layer                                       | A: previous layer's output W: previous layer's weights delta2                                                                    | delta: local gradient of neural j when j is hidden layer |
| `update_parameters(parameters,delta,cache,learning_rate=0.01)` | according to the forward and backward computation,update all weights in the network | parameters: python dictionary containing:W1,b1,W2,b2 delta:python tuple containing:delta1,delta2 cache:python dick learning_rate | parameters                                               
There are also many little funcions I implemented for show the results,due  to limited space, will not be showed here.
### Batch Learning
Due to memory limit,usually we can't take the total data into the model.Or noisy data happens when using online learning[1].Mini-batch is prefered when train a neural network.There are some tips in batch size selecting[2]:(1)A good default for batch size might be 32 (2)It is a good idea to review learning curves of model validation error against training time with different batch sizes when tuning the batch size.(3)Tune batch size and learning rate after tuning all other hyperparameters


## Reference
[1] [Standford CS230 Notes](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)    
[2] https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/


















<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>