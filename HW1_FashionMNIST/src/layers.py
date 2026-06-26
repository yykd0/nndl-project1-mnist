import numpy as np


class Linear:
    def __init__(self, in_dim, out_dim, rng):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -50, 50)
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class CrossEntropyLoss:
    def forward(self, logits, labels):
        labels = labels.reshape(-1).astype(np.int64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.labels = labels
        correct = self.probs[np.arange(labels.shape[0]), labels]
        return float(-np.mean(np.log(np.clip(correct, 1e-12, 1.0))))

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.labels.shape[0]), self.labels] -= 1.0
        grad /= self.labels.shape[0]
        return grad


def make_activation(name):
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    raise ValueError(f"Unsupported activation: {name}")
