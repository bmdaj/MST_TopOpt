from scipy.signal import convolve2d
import numpy as np

# -----------------------------------------------------------------------------------  
# AUXILIARY FUNCTIONS
# ----------------------------------------------------------------------------------- 

def conv2(x, y, mode='same'):
    """
    Python analogue to the Matlab conv2(A,B) function. Returns the two-dimensional convolution of matrices A and B.
    @ x: input matrix 1
    @ y: input matrix 2
    """
    return convolve2d(x, y, mode=mode)

def smooth_sign(x, beta):
    return np.tanh( beta  * x)

def resize_el_node(edofmat, A,  nElx, nEly):
    """
    Resizes the vector A from elements to nodes.
    @ edofmat: element dofs
    @ A: Material interpolation
    """

    A_nodes = np.zeros(((nEly+1)* (nElx+1)), dtype="complex128")
    
    for i in range(len(edofmat)):
        nodes = edofmat[i]
        val = A[i]
        A_nodes[nodes.astype(int)] += 0.25 * val # 4 nodes 

    return A_nodes