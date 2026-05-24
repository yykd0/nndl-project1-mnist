from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                else:
                    raise ValueError(f"Unsupported activation function: {act_func}")
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        if isinstance(param_list, dict):
            self.size_list = param_list['size_list']
            self.act_func = param_list['act_func']
            saved_params = param_list['params']
        else:
            self.size_list = param_list[0]
            self.act_func = param_list[1]
            saved_params = param_list[2:]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = saved_params[i]['W']
            layer.b = saved_params[i]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved_params[i].get('weight_decay', False)
            layer.weight_decay_lambda = saved_params[i].get('lambda', 1e-8)
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            else:
                raise ValueError(f"Unsupported activation function: {self.act_func}")
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = {
            'model_type': 'MLP',
            'size_list': self.size_list,
            'act_func': self.act_func,
            'params': []
        }
        for layer in self.layers:
            if layer.optimizable:
                param_list['params'].append({
                    'W' : layer.params['W'],
                    'b' : layer.params['b'],
                    'weight_decay' : layer.weight_decay,
                    'lambda' : layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_channels=8, kernel_size=5, hidden_dim=64, lambda_list=None):
        super().__init__()
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.lambda_list = lambda_list
        self.size_list = [784, 'conv', hidden_dim, 10]
        self.act_func = 'ReLU'
        self.layers = []
        self._build_layers()

    def _build_layers(self):
        conv = conv2D(1, self.conv_channels, self.kernel_size)
        fc1_in = self.conv_channels * 12 * 12
        fc1 = Linear(fc1_in, self.hidden_dim)
        fc2 = Linear(self.hidden_dim, 10)

        if self.lambda_list is not None:
            decay_layers = [conv, fc1, fc2]
            for layer, decay in zip(decay_layers, self.lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = decay

        self.layers = [
            conv,
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            fc1,
            ReLU(),
            fc2,
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            assert X.shape[1] == 28 * 28, "Flattened MNIST input must have 784 features."
            outputs = X.reshape(X.shape[0], 1, 28, 28)
        elif X.ndim == 3:
            outputs = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        else:
            outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        assert isinstance(param_list, dict) and param_list.get('model_type') == 'CNN'
        config = param_list['config']
        self.conv_channels = config['conv_channels']
        self.kernel_size = config['kernel_size']
        self.hidden_dim = config['hidden_dim']
        self.lambda_list = config.get('lambda_list')
        self.size_list = [784, 'conv', self.hidden_dim, 10]
        self.act_func = 'ReLU'
        self._build_layers()

        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved in zip(optimizable_layers, param_list['params']):
            layer.W = saved['W']
            layer.b = saved['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved.get('weight_decay', False)
            layer.weight_decay_lambda = saved.get('lambda', 1e-8)
        
    def save_model(self, save_path):
        param_list = {
            'model_type': 'CNN',
            'config': {
                'conv_channels': self.conv_channels,
                'kernel_size': self.kernel_size,
                'hidden_dim': self.hidden_dim,
                'lambda_list': self.lambda_list,
            },
            'params': []
        }
        for layer in self.layers:
            if layer.optimizable:
                param_list['params'].append({
                    'W' : layer.params['W'],
                    'b' : layer.params['b'],
                    'weight_decay' : layer.weight_decay,
                    'lambda' : layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
