import numpy as np

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = (input_size - self.W.shape[-1]) + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))

        
        for batch in range(batch_size):
            for out_channel in range(self.out_channels):
                for i in range(output_size):
                    if not(i+self.kernel_size > input_size):
                        val = np.sum(np.multiply(self.A[batch, :, i:i+self.kernel_size], self.W[out_channel, :, :])) 
                        Z[batch, out_channel, i] = val+ self.b[out_channel]
        return Z

    def backward(self, dLdZ):
        batch_size, out_channels, output_size = dLdZ.shape
        input_size = self.A.shape[2]

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        dLdA_padded = np.zeros(self.A.shape)
        for i in range(output_size):
            for j in range(out_channels):
                for k in range(self.in_channels):
                    self.dLdW[j, k, :] += np.sum(
                        self.A[:, k, i:i + self.kernel_size] * dLdZ[:, j, i:i + 1], axis=0)
                    dLdA_padded[:, k, i:i + self.kernel_size] += np.outer(
                        dLdZ[:, j, i], self.W[j, k, :])

        # Crop the padded gradient
        dLdA = dLdA_padded[:, :, self.kernel_size - 1:input_size]

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.padding = padding

        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = self.Downsample1d(stride)

    def forward(self, A):
        batch_size, in_channels, input_size = A.shape

        padded_A = np.pad(A, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant')
        conv_output = self.conv1d_stride1.forward(padded_A)
        Z = self.downsample1d.forward(conv_output)

        return Z

    def backward(self, dLdZ):
        dLdConv = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdConv)

        return dLdA
