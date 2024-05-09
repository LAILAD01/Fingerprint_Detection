import numpy as np

class Flatten:
    def forward(self, x):
        self.original_shape = x.shape  # Cache the original shape for backpropagation
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        # Reshape the gradient to the original input shape
        return d_out.reshape(self.original_shape)
