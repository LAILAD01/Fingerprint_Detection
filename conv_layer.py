import numpy as np

class ConvNet:
    def __init__(self, num_filters=8, filter_h=3, filter_w=3, stride=1):
    
        self.num_filters = num_filters
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.filters_weights = np.random.randn(num_filters, filter_h, filter_w) * 0.1
        self.bias = np.random.randn(num_filters, 1, 1) * 0.1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, d_out, x):
        dx = d_out.copy()
        dx[x <= 0] = 0
        return dx

    def zero_pad(self, x, pad):
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    def conv_forward(self, x, padding_mode='same'):
        n, h, w, _ = x.shape
        h_pad, w_pad = (self.filter_h - 1) // 2, (self.filter_w - 1) // 2
        if padding_mode == 'same':
            x_padded = self.zero_pad(x, h_pad)
        else:
            x_padded = x
        out_height = (h - self.filter_h + 2 * h_pad) // self.stride + 1
        out_width = (w - self.filter_w + 2 * w_pad) // self.stride + 1
        z = np.zeros((n, out_height, out_width, self.num_filters))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.filter_h
                w_end = w_start + self.filter_w
                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                for f in range(self.num_filters):
                    z[:, i, j, f] = np.sum(x_slice * self.filters_weights[f], axis=(1, 2, 3)) + self.bias[f]

        self.cache = (x, x_padded, z)
        return self.relu(z)

    def conv_backward(self, d_out, padding_mode):
        self.d_out = d_out
        x, x_padded, z = self.cache
        n, h, w, _ = x.shape
        h_pad, w_pad = (self.filter_h - 1) // 2, (self.filter_w - 1) // 2
        #dz: Gradient of the loss with respect to the output of this layer after ReLU.
        dz = self.relu_backward(d_out, z) 
        #dx: Gradient of the loss with respect to the input of this layer.
        dx_padded = np.zeros_like(x_padded)
        #dw: Gradient of the loss with respect to the weights of this layer.
        dw = np.zeros_like(self.filters_weights)
        #db: Gradient of the loss with respect to the biases of this layer.
        db = np.zeros_like(self.bias)

        for i in range((h - self.filter_h + 2 * h_pad) // self.stride + 1):
            for j in range((w - self.filter_w + 2 * w_pad) // self.stride + 1):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.filter_h
                w_end = w_start + self.filter_w

                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                for f in range(self.num_filters):
                    dw[f] += np.sum(x_slice * (dz[:, i:i+1, j:j+1, f:f+1]), axis=0)
                    db[f] += np.sum(dz[:, i:i+1, j:j+1, f:f+1], axis=(0, 1, 2))
                    dx_padded[:, h_start:h_end, w_start:w_end, :] += self.filters_weights[f] * dz[:, i:i+1, j:j+1, f:f+1]

        # Remove padding
        if padding_mode == 'same':
            dx = dx_padded[:, h_pad:-h_pad, w_pad:-w_pad, :]
        else:
            dx = dx_padded

        return dx, dw, db
