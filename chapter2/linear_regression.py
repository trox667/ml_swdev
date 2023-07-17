import numpy as np


def predict(X, w):
    return X * w


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X,Y,w)
        print(f'Iteration {i:4d} => Loss: {current_loss:.6f}')

        if loss(X,Y,w+lr) < current_loss:
            w += lr
        elif loss(X,Y,w-lr) < current_loss:
            w -= lr
        else:
            return w
    raise Exception(f'Could not converge within {iterations} iterations')

if __name__ == '__main__':

    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w = train(X,Y, iterations=1000, lr=0.01)
    print(f'\nw={w:.3f}')
    print(f'Prediction: x={20} => y={predict(20, w):.2f}')