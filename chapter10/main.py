import numpy as np


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1,1)


def prepend_bias(X):
    # array, index, value, columnwise
    return np.insert(X, 0, 1, axis=1)


def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat


def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]

def classify(X, w1, w2):
    y_hat = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print(f'Iteration: {iteration}, Loss: {training_loss:.6f}, Accuracy: {accuracy:.2f}')