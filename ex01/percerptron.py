import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, W):
        self.w = W  

    def predict(self, x):
        return 1.0 if np.dot(self.w.T, x) >= 0 else -1.0

    def fit(self, X, y):
        for epoch in range(50):
            y_calc = []
            for n in range(len(X)):
                y_hat = self.predict(X[n])
                if y_hat != y[n]:
                    self.w = self.w + np.multiply(y[n], X[n])
                y_calc.append(y_hat) 
        return self.w, y_calc


dataframe = pd.read_csv('data_xor.dat', delimiter="\s+", header=None, engine='python')
x1 = dataframe[0]
x2 = dataframe[1]
x_bias = np.ones(len(x1))
y = dataframe[2]
W = np.zeros(dataframe.shape[1])
X = [x for x in zip(x1,x2, x_bias)]
print(np.array(X).shape)

perceptron = Perceptron(W)
w_calc, y_calc = perceptron.fit(X,y)
print(w_calc)

def score(v1, v2):
    hits = 0
    for i in range(len(v1)):
        if(float(v1[i]) == float(v2[i])):
            hits += 1
    return hits/len(v1)
print(score(y, y_calc))

plot_step = 0.01

x_min, x_max = x1.min() - 0.2, x1.max() + 0.2
y_min, y_max = x2.min() - 0.2, x2.max() + 0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))

data = np.c_[xx.ravel(), yy.ravel()]

bias_col = np.ones(data.shape[0])

data = np.insert(data, 0, bias_col, axis=1)

perceptron_g = Perceptron(w_calc)

Z = np.array([perceptron_g.predict(x) for x in data])
Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

data = {'x1': x1, 'x2': x2}

plt.scatter('x1', 'x2', c=y, data=data)
plt.xlabel('entry x1')
plt.ylabel('entry x2')
plt.show()


