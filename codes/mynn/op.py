from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * scale
        self.b = np.zeros((1, out_dim), dtype=self.W.dtype)
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "Linear.backward called before forward."
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * self.kernel_size * self.kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1), dtype=self.W.dtype)
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded = None
        self.x_cols = None
        self.out_hw = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4, "conv2D expects input shape [batch, channels, H, W]."
        assert X.shape[1] == self.in_channels

        self.input = X
        if self.padding > 0:
            X_pad = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            X_pad = X
        self.input_padded = X_pad

        k = self.kernel_size
        windows = sliding_window_view(X_pad, (k, k), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        out_h, out_w = windows.shape[2], windows.shape[3]
        self.out_hw = (out_h, out_w)

        self.x_cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, self.in_channels * k * k)
        w_cols = self.W.reshape(self.out_channels, -1)
        out = self.x_cols @ w_cols.T
        out = out.reshape(X.shape[0], out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out + self.b

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.x_cols is not None, "conv2D.backward called before forward."
        batch_size = grads.shape[0]
        k = self.kernel_size
        out_h, out_w = self.out_hw

        dout_cols = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grads['W'] = (dout_cols.T @ self.x_cols).reshape(self.W.shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(self.b.shape)

        w_cols = self.W.reshape(self.out_channels, -1)
        dx_cols = dout_cols @ w_cols
        dx_windows = dx_cols.reshape(batch_size, out_h, out_w, self.in_channels, k, k)
        dx_windows = dx_windows.transpose(0, 3, 1, 2, 4, 5)

        dx_padded = np.zeros_like(self.input_padded)
        for i in range(k):
            row_slice = slice(i, i + out_h * self.stride, self.stride)
            for j in range(k):
                col_slice = slice(j, j + out_w * self.stride, self.stride)
                dx_padded[:, :, row_slice, col_slice] += dx_windows[:, :, :, :, i, j]

        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten all non-batch dimensions into one vector.
    """
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        assert self.input_shape is not None, "Flatten.backward called before forward."
        return grads.reshape(self.input_shape)

class MaxPool2D(Layer):
    """
    A simple max-pooling layer for CNN experiments.
    """
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.optimizable = False
        self.input = None
        self.max_mask = None
        self.out_hw = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert X.ndim == 4, "MaxPool2D expects input shape [batch, channels, H, W]."
        self.input = X
        k = self.kernel_size
        windows = sliding_window_view(X, (k, k), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        out = np.max(windows, axis=(-1, -2))
        self.out_hw = out.shape[2], out.shape[3]
        self.max_mask = windows == out[:, :, :, :, None, None]
        counts = np.sum(self.max_mask, axis=(-1, -2), keepdims=True)
        self.max_mask = self.max_mask / counts
        return out

    def backward(self, grads):
        assert self.input is not None and self.max_mask is not None, "MaxPool2D.backward called before forward."
        k = self.kernel_size
        out_h, out_w = self.out_hw
        dx = np.zeros_like(self.input)
        expanded = grads[:, :, :, :, None, None] * self.max_mask
        for i in range(k):
            row_slice = slice(i, i + out_h * self.stride, self.stride)
            for j in range(k):
                col_slice = slice(j, j + out_w * self.stride, self.stride)
                dx[:, :, row_slice, col_slice] += expanded[:, :, :, :, i, j]
        return dx

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.grads = None
        self.predicts = None
        self.labels = None
        self.probs = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = labels.astype(np.int64).reshape(-1)
        assert predicts.shape[0] == labels.shape[0]
        self.predicts = predicts
        self.labels = labels

        batch_size = predicts.shape[0]
        if self.has_softmax:
            self.probs = softmax(predicts)
            probs_for_loss = self.probs
        else:
            probs_for_loss = np.clip(predicts, 1e-12, 1.0)
            self.probs = probs_for_loss

        correct = probs_for_loss[np.arange(batch_size), labels]
        return -np.mean(np.log(np.clip(correct, 1e-12, 1.0)))
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.labels is not None, "Loss.backward called before forward."
        batch_size = self.probs.shape[0]
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:
            self.grads = -one_hot / np.clip(self.probs, 1e-12, 1.0) / batch_size
        # Then send the grads to model for back propagation
        if self.model is not None:
            return self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
