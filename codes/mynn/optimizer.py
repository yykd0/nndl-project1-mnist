from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

    def _optimizable_layers(self):
        return [layer for layer in self.model.layers if layer.optimizable]

    def _grad_with_decay(self, layer, key):
        grad = layer.grads[key]
        if grad is None:
            return None
        if layer.weight_decay and key == 'W':
            return grad + layer.weight_decay_lambda * layer.params[key]
        return grad


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self._optimizable_layers():
            for key in layer.params.keys():
                grad = self._grad_with_decay(layer, key)
                if grad is not None:
                    layer.params[key][...] -= self.init_lr * grad


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
    
    def step(self):
        for layer in self._optimizable_layers():
            layer_id = id(layer)
            if layer_id not in self.velocity:
                self.velocity[layer_id] = {
                    key: np.zeros_like(value) for key, value in layer.params.items()
                }
            for key in layer.params.keys():
                grad = self._grad_with_decay(layer, key)
                if grad is None:
                    continue
                self.velocity[layer_id][key] = self.mu * self.velocity[layer_id][key] - self.init_lr * grad
                layer.params[key][...] += self.velocity[layer_id][key]
