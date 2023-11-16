import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in_width)
        """
        self.batch_size, self.in_channels, self.in_width = A.shape

        # Reshape the input tensor into a 2D matrix
        Z = A.reshape(self.batch_size, self.in_channels * self.in_width)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels * in_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, in_width)
        """
        self.batch_size, in_channels_times_in_width = dLdZ.shape
        self.in_channels = in_channels_times_in_width // self.in_width

        # Reshape the gradient tensor back to the original shape
        dLdA = dLdZ.reshape(self.batch_size,self.in_channels, self.in_width)

        return dLdA
