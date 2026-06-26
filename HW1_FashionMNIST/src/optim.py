class SGD:
    def __init__(self, model, lr=0.1, weight_decay=0.0):
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def step(self):
        for param, grad in self.model.params_and_grads():
            update = grad
            if self.weight_decay > 0 and param.ndim > 1:
                update = update + self.weight_decay * param
            param[...] -= self.lr * update


class StepDecay:
    def __init__(self, optimizer, step_size=5, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def step(self, epoch):
        if self.step_size > 0 and epoch > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
