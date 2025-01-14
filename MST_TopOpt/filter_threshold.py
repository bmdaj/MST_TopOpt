import numpy as np
from .functions import conv2

class filter_threshold:
    """
    Class that that takes care of the filtering and thresholding.
    It helps ameliorate numerical issues ( i.e. pixel-by-pixel design variations) and introduce a weak sense of geometric scale.
    """
    def __init__(self, 
                 fR,
                 nElX,
                 nElY,
                 eta,
                 beta):
        """
        @ fR: Filtering radius.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ eta: parameter that controls threshold value.
        @ beta: parameter that controls the threshold sharpness.
        """
        
        self.fR = fR
        self.nElX = nElX
        self.nElY = nElY 
        self.eta = eta
        self.beta = beta
        self.filKer, self.filSca = self.density_filter_setup()
        

    def density_filter_setup(self):
        """
        Setup of a convolution-based filter.
        """
        dy, dx = np.meshgrid(np.arange(-np.ceil(self.fR)+1,np.ceil(self.fR)),
                             np.arange(-np.ceil(self.fR)+1,np.ceil(self.fR))) # we create a grid of values around fR for w(r)

        kernel  = np.maximum(np.zeros_like(dy, dtype="complex128"), self.fR*np.ones_like(dy, dtype="complex128")-np.sqrt(dx**2+dy**2)) # we create the kernel w(r)
        scaling = conv2(np.ones((self.nElY, self.nElX)), kernel, 'same') # we calculate the scaling by convolving 

        return kernel, scaling
    

    def density_filter(self, FilterScalingA, FilterScalingB , x, func):
        """
        Application of a convolution-based filter.
        """
        return conv2((x*func)/FilterScalingA, self.filKer, 'same')/FilterScalingB

    def threshold(self,x):
        """
        Application of a thresholding by means of a smoothed Heaviside-like function.
        """   
        x_out = (np.tanh(self.beta*self.eta)+np.tanh(self.beta*(x-self.eta)))/(np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta)))
    
        return x_out

    def deriv_threshold(self,x):
        """
        Derivative of the thresholding used in the sensitivity calculation.
        """ 

        x_out = (1-np.tanh(self.beta*(x-self.eta))**2)*self.beta/(np.tanh(self.beta*self.eta)+np.tanh(self.beta*(1-self.eta)))
    
        return x_out