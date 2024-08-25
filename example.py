import random
from micrograd.engine import Value
from micrograd.nn import mse, MLP, Linear, ReLU, Tanh

random.seed(0)

# data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# fmt: off
model = MLP([
    Linear(3, 4),
    ReLU(),
    Linear(4, 1),
    Tanh(),
])
# fmt: on

lr = 0.1
num_iter = 1000
show_iter_freq = 200
for i in range(1, num_iter + 1):
    y_preds = [model(x) for x in xs]
    y_preds = [y[0] for y in y_preds]
    loss = mse(y_preds, ys)

    # Zero grad
    model.zero_grad()

    # Backprop
    loss.backward()  # type: ignore

    for p in model.parameters():
        p.data += -lr * p.grad

    if i % show_iter_freq == 0:
        print(f"Epoch: {i}/{num_iter}, Train loss: {loss.data:.4f}")  # type: ignore


x = [Value(2.0), Value(3.0), Value(-1.0)]
y_preds = [model(x)[0]]
loss = mse(y_preds, [Value(ys[0])])
model.zero_grad()
loss.backward()
for xi in x:
    print(xi.data, xi.grad)
