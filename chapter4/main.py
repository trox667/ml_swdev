import numpy as np


def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print(f'Iteration {i:4d} => Loss: {loss(X,Y,w):.20f}')
        w -= gradient(X, Y, w) * lr
    return w


if __name__ == '__main__':

    # x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
    x1, x2, x3, y = np.loadtxt("life-expectancy-without-country-names.txt", skiprows=1, unpack=True)
    X = np.column_stack((np.ones(x1.size), x1, x2, x3))
    Y = y.reshape(-1, 1)

    w = train(X, Y, iterations=1000000, lr=0.0001)
    print(f'Weights: {w.T}')
    print(f'A few predictions:')
    for i in range(5):
        print(f'X[{i}] -> {predict(X[i], w)} (label: {Y[i]})')