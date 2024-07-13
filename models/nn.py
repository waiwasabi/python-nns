import numpy as np
import itertools as it


def pad(x):  # (batch, n, m) -> (batch, n+1, m)
    return np.pad(x, [(0, 0), (1, 0), (0, 0)], constant_values=1)


def trunc(x):  # (batch, n, m) -> (batch, n - 1, m)
    return x[:, 1:, :]


def T(x):  # (batch, n, m) -> (batch, m, n)
    return np.transpose(x, (0, 2, 1))


def zero(M):
    M[:, 0] = 0
    return M


def sigmoid(x):
    return np.piecewise(x, [x > 0], [
        lambda i: 1 / (1 + np.exp(-i)),
        lambda i: np.exp(i) / (1 + np.exp(i))
    ])


def cross_entropy(a, y):
    return -1/y.shape[1] * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))


class NN:
    def __init__(self, l, decay=0, act=sigmoid):
        self.l = l
        self.w = [np.random.normal(size=(y, x + 1))
                  for x, y in it.pairwise(self.l)]
        self.decay = decay
        self.act = act

    def cost(self, P, y):
        j = np.mean([cross_entropy(p, y_i) for p, y_i in zip(P, y)])
        return j + self.decay * sum(np.sum(w[:, 1:] ** 2) for w in self.w) / (2 * y.shape[0])

    def forward(self, x):
        a = []
        for w in self.w:
            a.append(x := pad(x))
            x = self.act(w @ x)
        a.append(x)
        return a

    def backward(self, a, y):
        d = [a[-1] - y]
        for w_i, a_i in zip(reversed(self.w[1:]), reversed(a[1:-1])):
            d.append(trunc(w_i.T @ d[-1] * a_i * (1 - a_i)))
        return reversed(d)

    def backprop(self, X, y):
        a = self.forward(X)
        d = self.backward(a, y)
        return [d_i @ T(a_i) for d_i, a_i in zip(d, a[:-1])]

    def update(self, grad, lr):
        self.w = [w - lr * g for w, g in zip(self.w, grad)]

    def step(self, X, y, lr):
        grad = self.backprop(X, y)
        reg = map(zero, [self.decay * w for w in self.w])

        grad = [(1 / X.shape[0]) * (np.sum(g, axis=0) +
                p) for g, p in zip(grad, reg)]

        self.update(grad, lr)
        return grad

    def train(self, X, y, lr, epochs, num_batches=1):
        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

        for _ in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            Xb, yb = map(lambda x: np.array_split(x, num_batches), (X, y))
            for Xb_i, yb_i in zip(Xb, yb):
                self.step(Xb_i, yb_i, lr)

    def predict(self, X):
        return self.forward(X)[-1]
