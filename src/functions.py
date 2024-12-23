#%%
from scipy.signal import convolve2d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt

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




def finite_diff(x, f, step, an_sens, nElx, nEly):
    """
    Function for finite difference check plus error callback.
    @ x: input variables
    @ fx: function evaluation f(x)
    @ step: step size
    @ an_sens: analytical sensitivities
    """

    fd_sens = np.zeros(len(x), dtype="complex128")
    x_new = x
    cost_function = f(x)

    for i in range(len(x)):
        x_new [i] += step
        cost_function_new = f(x_new)
        fd_sens [i] = (cost_function_new - cost_function) / step
        x_new [i] -= step

    import matplotlib.pyplot as plt


    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("science")
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 28})

    fig, ax = plt.subplots(3,1, figsize=(16,8))

    plot_array = np.array([an_sens, fd_sens, np.abs(np.real(an_sens) - np.real(fd_sens))])
    #titles = ["Analytical sensitivities, Finite-difference sensitivities, Absolute error"]
    print(np.shape(an_sens))
    print(np.shape(fd_sens))
    print(np.max(an_sens))
    print(np.max(fd_sens))

    for i in range(3):
        print(i)
    
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[i].imshow(np.reshape(np.real(plot_array[i]), (nEly, nElx)), aspect='auto', cmap='inferno', interpolation='none')
        fig.colorbar(im, cax=cax, orientation='vertical')
        #ax[i].set_title(titles[i])

    plt.show()

    raise("Stopping because of finite-difference check!")

#%%