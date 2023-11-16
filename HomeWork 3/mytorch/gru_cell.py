import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        # names below.
        # # #block 1
        term1 = np.dot(self.Wrh, h_prev_t) + self.brx            # (h,h) dot (h,) + (h,) = (h,)
        term2 = np.dot(self.Wrx, x) + self.brh            # (h,d) dot (d,) + (h,) = (h,)
        self.r = self.r_act.forward(term1 + term2) 

        #block2
        term1 = np.dot(self.Wzh, h_prev_t) + self.bzx            # (h,h) dot (h,) + (h,) = (h,)
        term2 = np.dot(self.Wzx, x) + self.bzh            # (h,d) dot (d,) + (h,) = (h,)
        self.z = self.r_act.forward(term1 + term2) 

        #block3
        term1 = np.dot(self.Wnx, x) + self.bnx            # (h,h) dot (h,) + (h,) = (h,)
        term2 = self.r * (np.dot(self.Wnh, h_prev_t) + self.bnh) # (h,) * (h,h) dot (h,) + (h,) = (h,)
        # save inner state to compute derivative in backprop easily
        self.n_state = np.dot(self.Wnh, h_prev_t) + self.bnh                       
        self.n = self.h_act.forward(term1 + term2)

        #block4 
        term1 = (1 - self.z) * self.n                     # (h,) * (h,) = (h,)
        term2 = self.z * h_prev_t                                # (h,) * (h,) = (h,)
        h_t = term1 + term2  

        
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
      

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # input_dim,=self.x.shape
        # hidden_dim,=self.hidden.shape

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        self.x = self.x.reshape(1,-1)
        self.hidden = self.hidden.reshape(1,-1)

        self.r = self.r.reshape(1,-1)
        self.z = self.z.reshape(1,-1)
        self.n = self.n.reshape(1,-1)

        # create 5 derivatives here itself for ease of troubleshooting
        dx = np.zeros_like(self.x) 
        dh_prev_t = np.zeros_like(self.hidden)
        dn = np.zeros_like(self.n) 
        dz = np.zeros_like(self.z)
        dr = np.zeros_like(self.z)

        #block4          
        dz += delta * (-self.n + self.hidden)      # (1,h) * (1,h) = (1,h)
        dn += delta * (1 - self.z)                 # (1,h) * (1,h) = (1,h)
        dh_prev_t += delta * self.z                       # (1,h) * (1,h) = (1,h)

        #block3
        grad_activ_n   = dn * (1-self.n**2)         # (1,h)
        r_grad_activ_n = grad_activ_n * self.r      # (1,h)

        self.dWnx += np.dot(grad_activ_n.T, self.x) # (h,1) dot (1,d) = (h,d)
        dx        += np.dot(grad_activ_n, self.Wnx) # (1,h) dot (h,d) = (1,d)
        self.dbnx += np.sum(grad_activ_n, axis=0)   # (1,h)
        dr        += grad_activ_n * self.n_state.T  # (1,h)

        self.dWnh += np.dot(r_grad_activ_n.T, self.hidden) # (h,1) dot (1,h) = (h,d)
        dh_prev_t        += np.dot(r_grad_activ_n, self.Wnh)      # (h,1) dot (1,h) = (h,d)
        self.dbnh += np.sum(r_grad_activ_n, axis=0)        # (1,h)

        #block2
        grad_activ_z = dz * self.z * (1-self.z)             # (1,h) * (1,h) * (1,h) = (1,h)
        
        dx        += np.dot(grad_activ_z, self.Wzx)         # (1,h) dot (h,d) = (1,d)
        self.dWzx += np.dot(grad_activ_z.T, self.x)         # (h,1) dot (1,d) = (h,d)
        self.dWzh += np.dot(grad_activ_z.T, self.hidden)    # (h,1) dot (1,d) = (h,d)
        dh_prev_t        += np.dot(grad_activ_z, self.Wzh)         # (1,h) dot (h,d) = (1,d)
        self.dbzx += np.sum(grad_activ_z, axis=0)           # (1,h)
        self.dbzh += np.sum(grad_activ_z, axis=0)           # (1,h)

        #block1
        grad_activ_r = dr * self.r * (1-self.r) # (h,1) dot (1,d) = (h,d)
        dx        += np.dot(grad_activ_r, self.Wrx) # (h,1) dot (1,d) = (h,d)
        self.dWrx += np.dot(grad_activ_r.T, self.x) # (h,1) dot (1,d) = (h,d)
        self.dWrh += np.dot(grad_activ_r.T, self.hidden) # (h,1) dot (1,d) = (h,d)
        dh_prev_t        += np.dot(grad_activ_r, self.Wrh) # (h,1) dot (1,d) = (h,d)
        self.dbrx += np.sum(grad_activ_r, axis=0) # (h,1) dot (1,d) = (h,d)
        self.dbrh += np.sum(grad_activ_r, axis=0) # (h,1) dot (1,d) = (h,d)


        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
        #raise NotImplementedError
        
        
