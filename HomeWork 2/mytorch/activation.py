import numpy as np
#import scipy
#from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):
        """
        Compute the sigmoid activation for each element in Z.
        """

        # Calculate the sigmoid activation using the sigmoid function: 1 / (1 + exp(-Z))
        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):
        """
        Compute the derivative of the loss with respect to Z using the chain rule.
        """

        # Calculate the derivative of the sigmoid function: sigmoid(Z) * (1 - sigmoid(Z))
        sigmoid_derivative = self.A * (1 - self.A)

        # Calculate the derivative of the loss with respect to Z using the chain rule: dLdZ = dLdA * sigmoid_derivative
        dLdZ = dLdA * sigmoid_derivative

        return dLdZ



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """

    def forward(self, Z):
        """
        Compute the hyperbolic tangent activation for each element in Z.
        """

        # Calculate the hyperbolic tangent activation using the tanh function: (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):
        """
        Compute the derivative of the loss with respect to Z using the chain rule.
        """

        # Calculate the derivative of the hyperbolic tangent function: 1 - tanh(Z)^2
        tanh_derivative = 1 - np.power(self.A, 2)

        # Calculate the derivative of the loss with respect to Z using the chain rule: dLdZ = dLdA * tanh_derivative
        dLdZ = dLdA * tanh_derivative

        return dLdZ



class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        """
        Compute the ReLU activation for each element in Z.
        """

        # Apply the ReLU activation: max(0, Z)
        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):
        """
        Compute the derivative of the loss with respect to Z using the chain rule.
        """

        # Calculate the derivative of the ReLU function: 1 if Z > 0, 0 otherwise
        relu_derivative = np.where(self.A > 0, 1, 0)

        # Calculate the derivative of the loss with respect to Z using the chain rule: dLdZ = dLdA * relu_derivative
        dLdZ = dLdA * relu_derivative

        return dLdZ

