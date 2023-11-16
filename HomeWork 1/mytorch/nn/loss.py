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
        # self.N, self.C = A.shape

        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (self.A - self.Y) ** 2  # TODO
        
        sse = np.sum(se) # TODO
        mse = sse /  (self.N * self.C) # TODO 

        return mse

    def backward(self):

        dLdA = 2* (self.A - self.Y) / (self.N * self.C)


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
        N = A.shape[0]  # TODO
        C = A.shape[1]  # TODO
        
        Ones_C = np.ones((C,), dtype='f')  # TODO
        Ones_N = np.ones((N,), dtype='f')  # TODO

        exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)  # TODO
        crossentropy = -np.sum(Y * np.log(self.softmax + 1e-10), axis=1)  # TODO
        sum_crossentropy =  np.sum(crossentropy)  # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):
        N = self.softmax.shape[0]
        dLdA=(self.softmax - self.Y) / N # TODO

        return dLdA
