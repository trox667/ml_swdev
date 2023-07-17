import numpy as np


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print(f'Iteration {i:4d} => Loss: {current_loss:.6f}')

        if loss(X, Y, w+lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w-lr, b) < current_loss:
            w -= lr
        if loss(X, Y, w, b+lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b-lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception(f'Could not converge within {iterations} iterations')


if __name__ == '__main__':

    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

    w, b = train(X, Y, iterations=10000, lr=0.01)
    print(f'\nw={w:.3f}, b={b:.3f}')
    print(f'Prediction: x={20} => y={predict(20, w, b):.2f}')
