import pickle

from .layers import Linear, make_activation


class ThreeLayerMLP:
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, activation="relu", seed=42):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation = activation.lower()
        self.seed = int(seed)
        rng = __import__("numpy").random.default_rng(seed)
        self.layers = [
            Linear(self.input_dim, self.hidden_dim, rng),
            make_activation(self.activation),
            Linear(self.hidden_dim, self.hidden_dim, rng),
            make_activation(self.activation),
            Linear(self.hidden_dim, self.output_dim, rng),
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        out = grad
        for layer in reversed(self.layers):
            out = layer.backward(out)
        return out

    def linear_layers(self):
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    def params_and_grads(self):
        for layer in self.linear_layers():
            yield from layer.params_and_grads()

    def first_layer_weights(self):
        return self.linear_layers()[0].W

    def state_dict(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "seed": self.seed,
            "weights": [(layer.W, layer.b) for layer in self.linear_layers()],
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(state["input_dim"], state["hidden_dim"], state["output_dim"], state["activation"], state.get("seed", 42))
        for layer, (W, b) in zip(model.linear_layers(), state["weights"]):
            layer.W[...] = W
            layer.b[...] = b
        return model
