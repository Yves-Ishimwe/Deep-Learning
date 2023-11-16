import numpy as np

class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # Number of samples
        self.C = A.shape[1]  # Number of classes

        se = np.square(A - Y)  # Squared error
        sse = np.sum(se)  # Sum of squared errors
        mse = sse / (self.N * self.C)  # Mean squared error

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)  # Derivative of MSE loss with respect to A

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]  # Number of samples
        C = A.shape[1]  # Number of classes

        Ones_C = np.ones((1, C), dtype='f')  # Array of ones with shape (1, C)
        Ones_N = np.ones((N, 1), dtype='f')  # Array of ones with shape (N, 1)

        self.softmax = np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)  # Softmax function
        crossentropy = -np.sum(Y * np.log(self.softmax))  # Cross entropy loss
        sum_crossentropy = np.sum(crossentropy)  # Sum of cross entropy losses
        L = sum_crossentropy / N  # Mean cross entropy loss

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.A.shape[0]  # Derivative of cross entropy loss with respect to A

        return dLdA