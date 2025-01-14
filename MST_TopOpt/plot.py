import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .functions import resize_el_node

def init_plot_params(fontsize):
    """
    Initialization of the plottings style used in different plotting routines.
    @ fontsize: Font size
    """
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": fontsize})


def plot_E_comp(dis, Ez):
    """
    Plots the electric field component for the whole simulation domain. For TE polarization we only have the E_z component.
    @ dis: The discretization class.
    """

    init_plot_params(22)

    fig, ax = plt.subplots(figsize=(6,3))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]

    im = ax.imshow(np.reshape(np.real(Ez), (dis.nodesY, dis.nodesX)), aspect='auto', cmap='seismic', interpolation='bilinear', extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical', label="$E_z$ (V/m)")

    # We also plot the contour of the permittivity

    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")

    ax.set_xlabel('$x$ (μm)')
    ax.set_ylabel('$y$ (μm)')

    plt.show()

def plot_H_comp(dis, comp):
    """
    Plots the magnetic field field components for the whole simulation domain. For TE polarization the magnetic field is (H_x, H_y, 0).
    @ dis: The discretization class.
    @ comp: The spatial component of the field, In this case: "x" or "y".
    """
    init_plot_params(22)

    fig, ax = plt.subplots(figsize=(6,3))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]

    if comp == "x":
        im = ax.imshow(np.real(dis.Hx), aspect='auto', cmap='seismic', interpolation='bilinear', extent=extent)
        fig.colorbar(im, cax=cax, orientation='vertical', label="$H_x$ (A/m)")


    if comp == "y":
        im = ax.imshow(np.real(dis.Hy), aspect='auto', cmap='seismic', interpolation='bilinear', extent=extent)
        fig.colorbar(im, cax=cax, orientation='vertical', label="$H_y$ (A/m)")


    # We also plot the contour of the permittivity
    
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")

    ax.set_xlabel('$x$ (μm)')
    ax.set_ylabel('$y$ (μm)')

    plt.show()


def plot_iteration(dis):
    """
    Plots the material interpolation and the electric field intensity for the whole simulation domain.
    Applied in each iteration of the optimization.
    @ dis: The discretization class.
    """
    init_plot_params(22)
    fig, ax = plt.subplots(1,2,figsize=(14,3))

    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]

    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    design_field =  np.reshape(np.real(dis.A), (dis.nodesY-1, dis.nodesX-1))
    ax[0].imshow(design_field, aspect='auto', cmap='binary', extent=extent)
    im = ax[1].imshow(np.reshape(np.real(dis.Ez*np.conj(dis.Ez)), (dis.nodesY, dis.nodesX)), extent=extent,aspect='auto', cmap='inferno', interpolation='bilinear')
    fig.colorbar(im, cax=cax, orientation='vertical', label="$|E_z|^2$(V$^2$/m$^2$)")

    # We also plot the contour of the permittivity
    
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax[1].contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")

    for axis in ax:
            axis.set_xlabel('$x$ (μm)')
    ax[0].set_ylabel('$y$ (μm)')
    plt.show()

def save_designs(nElX, nElY, scaling, dis, it_num, directory_opt):
    # This function needs to be revised and modified in this implementation.
    """
    Saves the plot of the design and the intensity field.
    @ nElX: Number of elements in the X axis.
    @ nElY: Number of elements in the Y axis.
    @ scaling: physical scaling of the physical problem, i.e. 1 nm.
    @ dis: The discretization class.
    @ it_num: iteration number in the optimization.
    @ directory_opt: directory to save optimization results.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(1,2,figsize=(24,8))
    extent = [-0.5*nElX*scaling * 1e9, 0.5*nElX*scaling * 1e9, -0.5*nElY*scaling * 1e9, 0.5*nElY*scaling * 1e9]

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im0 = ax[0].imshow(np.real(dis.dFPST+dis.dFPST_part), cmap='binary', vmax=1, vmin=0, extent=extent)
    I = np.real(dis.Ez*np.conj(dis.Ez))
    im =  ax[1].imshow(np.reshape(I, (nElY+1, nElX+1)), cmap='inferno', extent=extent, vmax=8)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # We also plot the contour of the permittivity

    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax[1].contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")

    for axis in ax:
            axis.set_xlabel('$x$ (nm)')
            axis.set_ylabel('$y$ (nm)')
    import os
    directory = directory_opt+"/design_history"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory_opt + "/design_history/design_it"+str(it_num)+".png")

def plot_it_history(FOM_list, constraint,it_num):
    """
    Plot the history of the optimization problem.
    @ FOM_list: List with the values of the FOM for the optimization iterations.
    @ constraint: List with the values of the connectivity constraint for the optimization iterations.
    @ it_num: iteration number in the optimization.
    """

    iterations = np.linspace(0,it_num-1, it_num)

    init_plot_params(22)
    fig, ax = plt.subplots(1,2,figsize=(20,3))

    ax[0].set_ylabel("FOM")
    ax[1].set_ylabel("Connectivity constraint")

    ax[0].scatter(iterations, FOM_list[:it_num], color='blue', edgecolor='black', s=100, alpha = 0.75)
    ax[0].plot(iterations, FOM_list[:it_num], color='blue',  alpha=0.5)

    ax[1].scatter(iterations, constraint [:it_num], color='red', edgecolor='black', s=100, alpha = 0.7)
    ax[1].plot(iterations, constraint [:it_num], color='red',  alpha=0.5)

    for axis in ax:
        axis.set_xlabel("Iteration number")

    plt.show()