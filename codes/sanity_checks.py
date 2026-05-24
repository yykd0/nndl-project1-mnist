import numpy as np

import mynn as nn


def check_linear_gradient():
    np.random.seed(1)
    layer = nn.op.Linear(4, 3)
    x = np.random.randn(5, 4)
    upstream = np.random.randn(5, 3)
    out = layer(x)
    layer.backward(upstream)
    analytic = layer.grads["W"][0, 0]
    eps = 1e-5
    original = layer.W[0, 0]
    layer.W[0, 0] = original + eps
    plus = np.sum(layer(x) * upstream)
    layer.W[0, 0] = original - eps
    minus = np.sum(layer(x) * upstream)
    layer.W[0, 0] = original
    numeric = (plus - minus) / (2 * eps)
    return abs(analytic - numeric)


def check_conv_gradient():
    np.random.seed(2)
    layer = nn.op.conv2D(1, 2, 3, stride=1, padding=1)
    x = np.random.randn(2, 1, 5, 5)
    upstream = np.random.randn(2, 2, 5, 5)
    layer.forward(x)
    layer.backward(upstream)
    analytic = layer.grads["W"][0, 0, 1, 1]
    eps = 1e-5
    original = layer.W[0, 0, 1, 1]
    layer.W[0, 0, 1, 1] = original + eps
    plus = np.sum(layer.forward(x) * upstream)
    layer.W[0, 0, 1, 1] = original - eps
    minus = np.sum(layer.forward(x) * upstream)
    layer.W[0, 0, 1, 1] = original
    numeric = (plus - minus) / (2 * eps)
    return abs(analytic - numeric)


def check_training_step():
    np.random.seed(3)
    model = nn.models.Model_MLP([4, 8, 3], "ReLU")
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=3)
    optimizer = nn.optimizer.SGD(init_lr=0.1, model=model)
    x = np.random.randn(16, 4)
    y = np.random.randint(0, 3, size=16)
    first = loss_fn(model(x), y)
    for _ in range(20):
        loss = loss_fn(model(x), y)
        loss_fn.backward()
        optimizer.step()
    last = loss_fn(model(x), y)
    return first, last


if __name__ == "__main__":
    linear_error = check_linear_gradient()
    conv_error = check_conv_gradient()
    first_loss, last_loss = check_training_step()
    print(f"linear_gradient_error={linear_error:.8e}")
    print(f"conv_gradient_error={conv_error:.8e}")
    print(f"training_loss_before={first_loss:.6f}")
    print(f"training_loss_after={last_loss:.6f}")
    assert linear_error < 1e-7
    assert conv_error < 1e-7
    assert last_loss < first_loss
    print("sanity_checks_passed")
