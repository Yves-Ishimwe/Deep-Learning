import numpy as np

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        self.in_features = in_features
        self.out_features = out_features
        self.dLdA = None
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features,))
        self.debug = debug

    def forward(self, A):
        self.A = A
        self.N = A.shape[0]
        Z = np.dot(A, self.W.T) + self.b.reshape(1, -1)
        return Z

    def backward(self, dLdZ):
        self.dLdA = np.dot(dLdZ, self.W)
        self.Ones = np.ones((self.N, 1))
        self.dLdW = np.dot(dLdZ.T, self.A)
        self.dLdb = np.dot(dLdZ.T, self.Ones)
        self.dLdW = self.dLdW
        self.dLdb = self.dLdb

        if self.debug:
            self.dLdA = self.dLdA

        return self.dLdA

# import numpy as np


# class Linear:

#     def __init__(self, in_features, out_features, debug=False):
#         """
#         Initialize the weights and biases with zeros
#         Checkout np.zeros function.
#         Read the writeup to identify the right shapes for all.
#         """
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dLdA= None
        
        
#         # Initialize weights W and bias b with zeros
#         self.W = np.zeros((out_features, in_features))
#         self.b = np.zeros((out_features,))
#         self.debug = debug

#     def forward(self, A):
#         """
#         :param A: Input to the linear layer with shape (N, C0)
#         :return: Output Z of linear layer with shape (N, C1)
#         Read the writeup for implementation details
#         """
#         self.A = A
#         self.N = A.shape[0]  # Batch size
#         Z = np.dot(A, self.W.T) + self.b.reshape(1, -1)
#         return Z
    
#     def backward(self, dLdZ):
# # Compute gradients
#         self.dLdA = np.dot(dLdZ, self.W)
#         self.Ones = np.ones((self.N,1))
#         self.dLdW = np.dot(dLdZ.T, self.A)
#         self.dLdb = np.dot(dLdZ.T, self.Ones)

#         # Store gradients for later use
#         self.dLdW = self.dLdW
#         self.dLdb = self.dLdb

#         if self.debug:
            
#             self.dLdA = dLdA

#         return dLdA
