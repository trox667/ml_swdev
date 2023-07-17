import numpy as np


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, 0) - Y))


def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print(f'Iteration {i:4d} => Loss: {loss(X,Y,w,0):.10f}')
        w -= gradient(X, Y, w) * lr
    return w, 0


if __name__ == '__main__':

    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w, b = train(X, Y, iterations=100, lr=0.001)
    print(f'\nw={w:.10f}, b={b:.3f}')
    # print(f'Prediction: x={20} => y={predict(20, w, b):.2f}')
