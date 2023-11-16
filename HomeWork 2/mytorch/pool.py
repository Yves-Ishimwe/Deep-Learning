import numpy as np


class MaxPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_size = self.kernel

        output_width = input_width - kernel_size + 1
        output_height = input_height - kernel_size + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[b, c, i, j] = np.max(A[b, c, i:i+kernel_size, j:j+kernel_size])

        return Z

    def backward(self, dLdZ):
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        kernel_size = self.kernel

        input_width = output_width + kernel_size - 1
        input_height = output_height + kernel_size - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        patch = dLdA[b, c, i:i+kernel_size, j:j+kernel_size]
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        dLdA[b, c, i:i+kernel_size, j:j+kernel_size] = mask * dLdZ[b, c, i, j]

        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        batch_size, in_channels, input_width, input_height = A.shape
        kernel_size = self.kernel

        output_width = input_width - kernel_size + 1
        output_height = input_height - kernel_size + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[b, c, i, j] = np.mean(A[b, c, i:i+kernel_size, j:j+kernel_size])

        return Z

    def backward(self, dLdZ):
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        kernel_size = self.kernel

        input_width = output_width + kernel_size - 1
        input_height = output_height + kernel_size - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        dLdA[b, c, i:i+kernel_size, j:j+kernel_size] = dLdZ[b, c, i, j] / (kernel_size * kernel_size)

        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        A_downsampled = self.downsample2d.forward(A)
        Z = self.maxpool2d_stride1.forward(A_downsampled)
        return Z

    def backward(self, dLdZ):
        dLdA_downsampled = self.maxpool2d_stride1.backward(dLdZ)
        dLdA = self.downsample2d.backward(dLdA_downsampled)
        return dLdA


class MeanPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        A_downsampled = self.downsample2d.forward(A)
        Z = self.meanpool2d_stride1.forward(A_downsampled)
        return Z

    def backward(self, dLdZ):
        dLdA_downsampled = self.meanpool2d_stride1.backward(dLdZ)
        dLdA = self.downsample2d.backward(dLdA_downsampled)
        return dLdA
