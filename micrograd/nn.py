import random
from .engine import Value
from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

    @abstractmethod
    def forward(self, x) -> Any:
        pass

    def __call__(self, x):
        return self.forward(x)


class Neuron(Module):
    def __init__(self, in_features, bias=False, requires_grad=True):
        self.w = [
            Value(random.uniform(-1, 1), requires_grad=requires_grad)
            for _ in range(in_features)
        ]
        if bias:
            self.b = Value(random.uniform(-1, 1), requires_grad=requires_grad)
        else:
            self.b = Value(0.0, requires_grad=requires_grad)

    def forward(self, x):
        # w @ x + b
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out

    def parameters(self):
        return self.w + [self.b]


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class ReLU(Module):
    def forward(self, x):
        out = [xi.relu() for xi in x]
        return out


class Tanh(Module):
    def forward(self, x):
        out = [xi.tanh() for xi in x]
        return out


class Sigmoid(Module):
    def forward(self, x):
        out = [xi.sigmoid() for xi in x]
        return out


class MLP(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse(y_preds, y_trues):
    N = len(y_preds)
    return sum((y_pred - y_true) ** 2 for y_pred, y_true in zip(y_preds, y_trues)) / N


__all__ = ["Module", "Linear", "ReLU", "Tanh", "Sigmoid", "MLP", "mse"]
if __name__ == "__main__":
    random.seed(0)
    x = [2.0, 3.0]
    x = [Value(xi) for xi in x]
    # fmt: off
    model = MLP([
        Linear(2, 5),
        ReLU(),
        Linear(5, 5),
        ReLU(),
        Linear(5, 1),
        Sigmoid()
    ])

    print(model(x))
    print(Sigmoid()([Value(100)]))
    for xi in x:
        print(xi.grad)
