import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1

        Z = np.zeros((batch_size, in_channels, output_width))  # Initialize Z with zeros
        Z[:, :, ::self.upsampling_factor] = A


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width-1) // self.upsampling_factor+1

        dLdA = np.zeros((batch_size, in_channels, input_width))  # Initialize dLdA with zeros

        # Copy the non-zero gradients from dLdZ to the corresponding positions in dLdA
        dLdA += dLdZ[:, :, ::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        self.input_width = input_width
        output_width = input_width // self.downsampling_factor
        Z = np.zeros((batch_size, in_channels, output_width))  # Initialize Z with zeros

        Z = A[:, :, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        input_width = output_width * self.downsampling_factor
        input_width = self.input_width 

        dLdA = np.zeros((batch_size, in_channels, input_width))
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size, in_channels, input_height, input_width = A.shape
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))  # Initialize Z with zeros

        # Copy the original data to every k-th position in Z
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = (output_height-1) // self.upsampling_factor+1
        input_width = (output_width-1) // self.upsampling_factor+1
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))  # Initialize dLdA with zeros
        dLdA[:, :, :output_height, :output_width] = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A_size = A.shape
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        array2D = [[1] + [0]*(self.downsampling_factor-1)] + [[0]*self.downsampling_factor]*(self.downsampling_factor-1)


        dLdA = np.kron(dLdZ, array2D) # TODO
        dLdA = dLdA[:, :, :self.A_size[-2], :self.A_size[-1]]  #TODO

        return dLdA