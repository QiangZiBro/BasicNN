"""
Using perceptron convergence theroem to solve basic binary classfication problem
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    """
    A perceptron model with:
        N instance
        n features
        2 classes for {-1, +1}
        w weights
        b bias
    Just give initial partition parameters,and number of instances,
    the program will use gradient descent to compute decision bound.
    """
    def __init__(self,initial_weights,initial_bias,N):
        self.dataSet,self.parameters= self.initializeData(initial_weights,initial_bias,N)

        self.result = None
    def initializeData(self,w,b,N):
        """
        args:
            w: a (n,1) shape initialized weight for coefficient of superplace :`wx+b=0`
            b: bias for superplane,also a scalar
            N: number of instances or points.
        return:
            dataset:the last 500 rows is the line seperating the two classes
        """
        w = np.array(w)
        numFeatures = len(w)
        parameters = {"w": w, "b": b, "N":N,"n":numFeatures}

        x = np.random.rand(N, numFeatures) * 20  #随机产生N个数据的数据集
        cls = np.sign(np.sum(w*x,axis=1)+b)    #用标准线 w*x+b=0进行分类
        dataSet = np.column_stack((x,cls))    #我们希望的数据集格式: x0, x1, ..., xn-1, y
        return dataSet,parameters

    def train(self,dataSet,parameters,learning_rate = 1):
        b = parameters["b"]
        n = parameters["n"]
        N = parameters["N"]

        w = np.ones((1, n))  # row vector
        b = 0
        i = 0

        while i < N:
            yi = dataSet[i,-1]
            xi = dataSet[i,0:-1]
            if yi * (np.sum(w * dataSet[i,0:-1],)+ b)< 0:
                w += learning_rate * yi * xi
                b += learning_rate * yi
                i = 0 #[BUG FIXED]
            else:
                i += 1

        cache = {"w": w, "b": b, "N": N, "n": n}

        return cache

    def show(self,dataSet,parameters):


        w0 = self.parameters["w"]
        b0 = self.parameters["b"]
        w = parameters["w"]
        b = parameters["b"]
        n = parameters["n"]
        N = parameters["N"]

        labels = ['classOne', 'classTow', 'modelLine', 'standardLine']
        markers = ['o', 'x', '.', '.']
        colors = ['r', 'g', 'b', 'y']

        fig = plt.figure()
        if n == 2:
            ax = fig.add_subplot(1, 1, 1)
            # 先将基本点画出来
            for i, cls in enumerate([-1, 1]):
                idx = np.where(dataSet[:, -1] == cls)
                ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker=markers[i], color=colors[i], label=labels[i], s=25)
            # 画拟合出来的线
            x1 = np.linspace(0, 20, 500)  # 创建分类线上的点，以点构线。 (500,1)
            x2 = -w[...,0] / w[...,1] * x1 - b / w[...,1]  # (500,1)
            ax.scatter(x1, x2, marker=markers[2], color=colors[2], label=labels[2], s=25)

            # 画实际线
            x1 = np.linspace(0, 20, 500)  # 创建分类线上的点，以点构线。 (500,1)
            x2 = -w0[0] / w0[1] * x1 - b0 / w0[1]  # (500,1)
            ax.scatter(x1, x2, marker=markers[3], color=colors[3], label=labels[3], s=25)

        elif n == 3:
            #ax = fig.add_subplot(111, projection='3d')
            ax = fig.gca(projection='3d')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            for i,cls in enumerate([-1,1]):
                idx = np.where(dataSet[:, -1] == cls)
                ax.scatter(dataSet[idx, 0], dataSet[idx, 1], dataSet[idx,2],marker=markers[i], color=colors[i], label=labels[i], s=25)
            # 画拟合出来的面
            x = np.linspace(0, 20, 40)
            y = np.linspace(0, 20, 40)
            X, Y = np.meshgrid(x, y)
            Z = -w[...,0] / w[...,2] * X - w[...,1] / w[...,2] * Y - b / w[...,2]
            ax.plot_surface(X, Y, Z, color="purple")
            # 画实际面
            X, Y = np.meshgrid(x, y)
            Z = -w0[0] / w0[2] * X - w0[1] / w0[2] * Y - b / w0[2]
            ax.plot_surface(X, Y, Z, color="blue")
        else:
            print("high dimension(n > 3)")

        plt.show()

    def fit(self):
        result = self.train(self.dataSet,self.parameters)
        self.show(self.dataSet, result)
        self.result = result
        return result
    def predict(self,x):
        w = self.result["w"]
        b = self.result["b"]
        if np.sum(w * x)+ b <= 0:
            return -1
        else:
            return 1

if __name__ == "__main__":
    #model = Perceptron([-6, 6], 7, 200)
    # or:
    model = Perceptron([-6, 6, 6], 7, 200)
    model.fit()
