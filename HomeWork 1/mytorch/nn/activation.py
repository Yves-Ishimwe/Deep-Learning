import numpy as np
import scipy
import scipy.special


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
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A**2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = (self.A > 0).astype(float)
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = 0.5*Z*(1 + scipy.special.erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        dAdZ = 0.5*(1 + scipy.special.erf(self.Z / np.sqrt(2))) + \
                    (self.Z/np.sqrt(2 * np.pi)) * np.exp(-(self.Z)**2 / 2)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))

        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)  # TODO

        return self.A

 

    def backward(self, dLdA):
        
        

        # Calculate the batch size and number of features
        N = dLdA.shape[0] # TODO
        C = dLdA.shape[1] # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(self.A) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = -np.outer(self.A[i],self.A[i]) # TODO
            J[np.diag_indices(C)]=self.A[i]*(1-self.A[i])
            dLdZ[i]=np.sum(J*dLdA[i],axis=1)

            # # Fill the Jacobian matrix according to the conditions described in the writeup
            # for m in range(C):
            #     for n in range(C):
            #         J[m,n] = self.A[i, m] * (1 - self.A[i, n]) # TODO
            

            # # Calculate the derivative of the loss with respect to the i-th input
            # dLdZ[i,:] = np.dot(J, dLdA[i, :]) # TODO
            
        return dLdZ