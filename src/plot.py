import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import resize_el_node

def init_plot_params(fontsize):
    """
    Initialization of the plottings style used in different plotting routines.
    @ fontsize: Font size
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("science")
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": fontsize})


def plot_intensity(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(22)

    fig, ax = plt.subplots(figsize=(18,8))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    print(np.max(np.real(dis.Ez)))
    #im = ax.imshow(np.reshape(np.imag(dis.Ez), (dis.nodesY, dis.nodesX)), aspect='auto', cmap='inferno', interpolation='none')
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    im = ax.imshow(np.reshape(np.real(dis.Ez), (dis.nodesY, dis.nodesX)), aspect='auto', extent=extent, cmap='inferno', interpolation='bilinear')
    #im = ax.imshow(np.reshape(np.real(dis.Ez*np.conj(dis.Ez)), (dis.nodesY, dis.nodesX)), aspect='auto', extent=extent, cmap='inferno', interpolation='bilinear')
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (\\textmu m)')
    ax.set_ylabel('$y$ (\\textmu m)')

    #print(np.max(np.real(dis.Ez*np.conj(dis.Ez))))

    plt.show()

def plot_H_comp(dis, comp):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(16,8))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    extent = [-0.5*dis.nElx*dis.scaling * 1e9, 0.5*dis.nElx*dis.scaling * 1e9, -0.5*dis.nEly*dis.scaling * 1e9, 0.5*dis.nEly*dis.scaling * 1e9]

    if comp == "x":
        im = ax.imshow(np.real(dis.Hx), aspect='auto', cmap='inferno', interpolation='bilinear', extent=extent)
        print(np.max(np.real(dis.Hx)))

    if comp == "y":
        im = ax.imshow(np.real(dis.Hy), aspect='auto', cmap='inferno', interpolation='bilinear', extent=extent)
        print(np.max(np.real(dis.Hy)))

    
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    plt.show()

def plot_heat(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(16)
    fig, ax = plt.subplots(figsize=(14,12))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    print(np.max(np.real(dis.T)))
    #im = ax.imshow(np.reshape(np.imag(dis.Ez), (dis.nodesY, dis.nodesX)), aspect='auto', cmap='inferno', interpolation='none')
    extent = [-0.5*dis.nElx*dis.scaling * 1e9, 0.5*dis.nElx*dis.scaling * 1e9, -0.5*dis.nEly*dis.scaling * 1e9, 0.5*dis.nEly*dis.scaling * 1e9]
    im = ax.imshow(np.reshape(np.real(dis.T), (dis.nodesY, dis.nodesX)), extent=extent, cmap='inferno', interpolation='none')
    eps = resize_el_node(dis.edofMat, np.real(dis.dFPST).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    #print(np.max(np.real(dis.Ez*np.conj(dis.Ez))))

    plt.show()

def plot_mi(dis):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(16,8))

    extent = [-0.5*dis.nElx*dis.scaling * 1e9, 0.5*dis.nElx*dis.scaling * 1e9, -0.5*dis.nEly*dis.scaling * 1e9, 0.5*dis.nEly*dis.scaling * 1e9]

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.A), (dis.nodesY-1, dis.nodesX-1)), aspect='auto', cmap='inferno', extent=extent, origin="upper")
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    plt.show()

def plot_mi_heat(dis):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(2,1,figsize=(16,8))

    extent = [-0.5*dis.nElx*dis.scaling * 1e9, 0.5*dis.nElx*dis.scaling * 1e9, -0.5*dis.nEly*dis.scaling * 1e9, 0.5*dis.nEly*dis.scaling * 1e9]

    
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    im0 = ax[0].imshow(np.reshape(np.real(dis.A_C), (dis.nodesY-1, dis.nodesX-1)), aspect='auto', cmap='inferno', extent=extent, origin="upper")
    fig.colorbar(im0, cax=cax0, orientation='vertical')

    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im1 = ax[1].imshow(np.reshape(np.real(dis.A_S), (dis.nodesY-1, dis.nodesX-1)), aspect='auto', cmap='inferno', extent=extent, origin="upper")
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    for axis in ax:

        axis.set_xlabel('$x$ (nm)')
        axis.set_ylabel('$y$ (nm)')

    plt.show()

def plot_iteration(dis):
    """
    Plots the material interpolation and the electric field intensity for the whole simulation domain.
    Applied in each iteration of the optimization.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(1,2,figsize=(20,4))

    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]

    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    design_field =  np.reshape(np.real(dis.A), (dis.nodesY-1, dis.nodesX-1))
    ax[0].imshow(design_field, aspect='auto', cmap='binary', extent=extent,)
    im = ax[1].imshow(np.reshape(np.real(dis.Ez*np.conj(dis.Ez)), (dis.nodesY, dis.nodesX)), extent=extent,aspect='auto', cmap='inferno', interpolation='bilinear')
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    #ax[0].imshow(np.real(eps), aspect='auto', cmap='binary')
    ax[1].contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")
    fig.colorbar(im, cax=cax, orientation='vertical')
    for axis in ax:
            axis.set_xlabel('$x$ (\\textmu m)')
    ax[0].set_ylabel('$y$ (\\textmu m)')
    plt.show()

def plot_sens(dis, sens):
    """
    Plots the sensitivities for the design region.
    """
    init_plot_params(16)
    fig, ax = plt.subplots(figsize=(14,12))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(sens), (len(dis.dVElmIdx[0]), len(dis.dVElmIdx[1]))), cmap='inferno')
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    plt.show()

def plot_sens_heat(dis, sens):
    """
    Plots the sensitivities for the design region.
    """
    init_plot_params(16)
    fig, ax = plt.subplots(figsize=(14,12))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(sens), (dis.nEly, dis.nElx)), cmap='inferno')
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    plt.show()

def plot_sens_part(dis, sens):
    """
    Plots the sensitivities for the design region.
    """
    init_plot_params(16)
    fig, ax = plt.subplots(figsize=(14,12))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(sens), (len(dis.dVElmIdx_part[0]), len(dis.dVElmIdx_part[1]))), cmap='inferno')
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')

    plt.show()

def save_designs(nElX, nElY, scaling, dis, it_num, directory_opt):

    init_plot_params(28)

    fig, ax = plt.subplots(1,2,figsize=(24,8))
    extent = [-0.5*nElX*scaling * 1e9, 0.5*nElX*scaling * 1e9, -0.5*nElY*scaling * 1e9, 0.5*nElY*scaling * 1e9]

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im0 = ax[0].imshow(np.real(dis.dFPST+dis.dFPST_part), cmap='binary', vmax=1, vmin=0, extent=extent)
    I = np.real(dis.Ez*np.conj(dis.Ez))
    im =  ax[1].imshow(np.reshape(I, (nElY+1, nElX+1)), cmap='inferno', extent=extent, vmax=8)
    eps = resize_el_node(dis.edofMat, np.real(dis.A).flatten(), dis.nElx, dis.nEly)
    eps = np.reshape(eps, (dis.nodesY, dis.nodesX))
    ax[1].contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent, origin="upper")

    fig.colorbar(im, cax=cax, orientation='vertical')


    for axis in ax:
            axis.set_xlabel('$x$ (nm)')
            axis.set_ylabel('$y$ (nm)')
    import os
    directory = directory_opt+"/design_history"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory_opt + "/design_history/design_it"+str(it_num)+".png")

def plot_it_history(maxItr, FOM_list, constraint_1, constraint_2, it_num, save, dir):

    iterations = np.linspace(0,it_num-1, it_num)

    init_plot_params(24)
    fig, ax = plt.subplots(1,2,figsize=(16,6))

    ax[0].set_ylabel("FOM")
    ax[1].set_ylabel("Connectivity constraint")

    ax[0].scatter(iterations, FOM_list [:it_num], color='blue')
    ax[0].plot(iterations, FOM_list[:it_num], color='blue',  alpha=0.5)

    ax[1].scatter(iterations, constraint_1 [:it_num], color='red')
    ax[1].plot(iterations, constraint_1 [:it_num], color='red',  alpha=0.5, label="Metalens")

    ax[1].scatter(iterations, constraint_2 [:it_num], color='green')
    ax[1].plot(iterations, constraint_2 [:it_num], color='green',  alpha=0.5, label="Particle")

    ax[1].legend(frameon=True, fontsize=20)
    ax[0].legend(frameon=True, fontsize=20)
        
    for axis in ax:
        axis.set_xlabel("Iteration number")

    if save:
        fig.savefig(dir+"/iteration_history.svg")

    plt.show()