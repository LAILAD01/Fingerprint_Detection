import numpy as np

class MaxPooling:
    def __init__(self, pool_h=2, pool_w=2, stride=2):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride

    def forward(self, x):
        n, h, w, c = x.shape
        out_h = (h - self.pool_h) // self.stride + 1
        out_w = (w - self.pool_w) // self.stride + 1
        out = np.zeros((n, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_h
                w_end = w_start + self.pool_w
                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(x_slice, axis=(1, 2))

        self.cache = (x, out)
        return out

    def backward(self, d_out):
        x, out = self.cache
        n, h, w, c = x.shape
        out_h, out_w = d_out.shape[1], d_out.shape[2]
        dx = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_h
                w_end = w_start + self.pool_w
                for n_ex in range(n):
                    for c_ex in range(c):
                        x_slice = x[n_ex, h_start:h_end, w_start:w_end, c_ex]
                        mask = (x_slice == np.max(x_slice))
                        dx[n_ex, h_start:h_end, w_start:w_end, c_ex] += d_out[n_ex, i, j, c_ex] * mask

        return dx
