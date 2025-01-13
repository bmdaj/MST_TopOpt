import numpy as np
import matplotlib.pyplot as plt
from functions import resize_el_node, smooth_sign

# EVENTUALLY CHANGE CODE TO HAVE AN OPTOMECHANICS CLASS

def calc_MST(Ex,Ey,Ez,Hx,Hy,Hz):

    """
    Calculates the Maxwell Stress Tensor (MST) for an electromagnetic field.
    @ Ex: Ex field
    @ Ey: Ey field
    @ Ez: Ez field
    @ Hx: Hx field
    @ Hy: Hy field
    @ Hz: Hz field
    """    

    eps_0 = 8.854187816292039E-12
    mu_0 = 1.2566370616244444E-6

    Ex_2 = Ex * np.conj(Ex)
    Ey_2 = Ey * np.conj(Ey)
    Ez_2 = Ez * np.conj(Ez)

    E_2 = Ex_2 + Ey_2 + Ez_2

    Hx_2 = Hx * np.conj(Hx)
    Hy_2 = Hy * np.conj(Hy)
    Hz_2 = Hz * np.conj(Hz)

    H_2 =  Hx_2 + Hy_2 + Hz_2

    Txx = eps_0 * (Ex_2 - 0.5*E_2) + mu_0 * (Hx_2 - 0.5*H_2)
    Tyy = eps_0 * (Ey_2 - 0.5*E_2) + mu_0 * (Hy_2 - 0.5*H_2)
    Tzz = eps_0 * (Ez_2 - 0.5*E_2) + mu_0 * (Hz_2 - 0.5*H_2)

    Txy = eps_0 * (Ex * np.conj(Ey)) + mu_0 * (Hx * np.conj(Hy))
    Tyx = eps_0 * (Ey * np.conj(Ex)) + mu_0 * (Hy * np.conj(Hx))

    Txz = eps_0 * (Ex * np.conj(Ez)) + mu_0 * (Hx * np.conj(Hz))
    Tzx = eps_0 * (Ez * np.conj(Ex)) + mu_0 * (Hz * np.conj(Hx))

    Tyz = eps_0 * (Ey * np.conj(Ez)) + mu_0 * (Hy * np.conj(Hz))
    Tzy = eps_0 * (Ez * np.conj(Ey)) + mu_0 * (Hz * np.conj(Hy))


    T = 0.5*np.real(np.array([[Txx,Txy,Txz],[Tyx,Tyy,Tyz],[Tzx,Tzy,Tzz]])) # 0.5*np.real() like in COMSOL for cycle-averaging

    T_px = 0.5*np.real(np.array([Txx, Tyx, Tzx]))
    T_py = 0.5*np.real(np.array([Txy, Tyy, Tzy]))

    return T, T_px, T_py

def calc_dMSTdEz(alphaz, betax, betay, betaxy, betayx):

    """
    Calculates derivative of the Maxwell Stress Tensor (MST) for an electromagnetic field.
    """    

    eps_0 = 8.854187816292039E-12
    mu_0 = 1.2566370616244444E-6


    Txx = -eps_0 * alphaz + mu_0 * (betax-betay)
    Tyy = -eps_0 * alphaz + mu_0 * (betay-betax)
    Tzz = eps_0 * alphaz - mu_0 * (betax+betay)

    Txy = 2* mu_0 * betayx
    Tyx = 2* mu_0 * betaxy

    Txz = np.zeros_like(Txx)
    Tzx = np.zeros_like(Txx)

    Tyz = np.zeros_like(Txx)
    Tzy = np.zeros_like(Txx)

    T = 0.25*np.array([[Txx,Txy,Txz],[Tyx,Tyy,Tyz],[Tzx,Tzy,Tzz]]) # 0.5*np.real() like in COMSOL for cycle-averaging

    # return tensor and projections as in COMSOL

    T_px = 0.25*np.array([Txx, Txy, Txz])
    T_py = 0.25*np.array([Tyx, Tyy, Tyz])

    return T, T_px, T_py

def calc_F(T_p, b_idx_x, b_idx_y, b_n_x, b_n_y, scaling):

    """
    Calculates the force on the particle.
    @ T_p: Maxwell stress tensor projection onto axis
    @ b_idx_x: Index of the boundary elements for the nx components
    @ b_idx_y: Index of the boundary elements for the ny components
    @ b_n_x: Normal of the boundary elements for the x components
    @ b_n_y: Normal of the boundary elements for the y components
    @ scaling: Scaling factor
    """

    T_p_bx = T_p[0, b_idx_x] # Projection of the X component of the stress tensor onto the boundary elements
    T_p_by = T_p[1, b_idx_y] # Projection of the Y component of the stress tensor onto the boundary elements

    F = (np.sum(np.real(T_p_bx)*b_n_x) +  np.sum(np.real(T_p_by)*b_n_y))* scaling

    return np.real(F)

def calc_P(T_p, b_idx_x, b_idx_y, b_n_x, b_n_y, scaling):

    """
    Calculates the force on the particle.
    @ T_p: Maxwell stress tensor projection onto axis
    @ b_idx_x: Index of the boundary elements for the nx components
    @ b_idx_y: Index of the boundary elements for the ny components
    @ b_n_x: Normal of the boundary elements for the x components
    @ b_n_y: Normal of the boundary elements for the y components
    @ scaling: Scaling factor
    """

    T_p_bx = T_p[0, b_idx_x] # Projection of the X component of the stress tensor onto the boundary elements
    T_p_by = T_p[1, b_idx_y] # Projection of the Y component of the stress tensor onto the boundary elements

    F = (np.sum(np.real(T_p_bx)) +  np.sum(np.real(T_p_by)))

    return F
    
def find_boundaries(A, edofMat, nElx, nEly):

    """
    Finds the indexes of the boundary elements.
    @ A: Material interpolation
    """

    A_node = resize_el_node(edofMat, A.flatten(), nElx, nEly)

    A_node = np.reshape(A_node, (nEly+1, nElx+1))

    grad_A_node = np.gradient(A_node, edge_order=2)

    grad_A_node_y = grad_A_node[0]
    grad_A_node_x = grad_A_node[1]
    
    
    condition_x = (A_node == 0.0) & (np.abs(grad_A_node_x) > 0.1) 
    condition_y = (A_node == 0.0) & (np.abs(grad_A_node_y) > 0.1) 

    indexes_x = np.where(condition_x)
    indexes_y = np.where(condition_y)


    # Combine indexes_x and indexes_y into one array

    new = np.zeros_like(A_node)

    #new[indexes_x] = 1.0
    #new[indexes_y] = 1.0

    normals_x = -np.sign(grad_A_node_x[indexes_x])
    normals_y = -np.sign(grad_A_node_y[indexes_y])

    #new = A_node

    #new[indexes_x] = normals_x
    #new[indexes_y] = normals_y

    #plt.rcParams.update(plt.rcParamsDefault)
    #plt.style.use("science")
    #import matplotlib as mpl
    #from mpl_toolkits.axes_grid1 import make_axes_locatable

    #mpl.rcParams.update({"font.size": 28})


    #fig, ax = plt.subplots(figsize=(16,8))
    
    #im = ax.imshow(new, aspect='auto', cmap='inferno', interpolation='none', origin='lower')
    #ax.set_xlabel('$X$ (nm)')
    #ax.set_ylabel('$Y$ (nm)')

    #plt.show()

    #raise()

    return indexes_x, indexes_y, normals_x, normals_y

def find_boundaries_gray(A, edofMat, nElx, nEly):

    """
    Finds the indexes of the boundary elements.
    @ A: Material interpolation
    """

    #grad_A = np.gradient(A, edge_order=2)
    #grad_A_y = grad_A[0]
    #grad_A_x = grad_A[1]
    #grad_A_node_y = np.reshape(resize_el_node(edofMat, grad_A_y.flatten(), nElx, nEly), (nEly+1, nElx+1))
    #grad_A_node_x = np.reshape(resize_el_node(edofMat, grad_A_x.flatten(), nElx, nEly), (nEly+1, nElx+1))


    A_node = resize_el_node(edofMat, A.flatten(), nElx, nEly)

    A_node = np.reshape(A_node, (nEly+1, nElx+1))

    grad_A_node = np.gradient(A_node, edge_order=2)
    

    grad_A_node_y = grad_A_node[0] * 2*(1.0-A_node) 
    grad_A_node_x = grad_A_node[1] * 2*(1.0-A_node) 

    # Compute the gradient along the y-axis (axis=0)
    
    eps = 1E-3
    
    condition_x = (np.abs(grad_A_node_x) > eps) 
    condition_y = (np.abs(grad_A_node_y) > eps)

    #condition_x = (np.abs(grad_A_node_x) == np.max(np.abs(grad_A_node_x))) 
    #condition_y = (np.abs(grad_A_node_y) == np.max(np.abs(grad_A_node_y))) 

    indexes_x = np.where(condition_x)
    indexes_y = np.where(condition_y)


    # Combine indexes_x and indexes_y into one array

    new = np.zeros_like(grad_A_node_x)

    #new[indexes_x] = 1.0
    #new[indexes_y] = 1.0

    normals_x = grad_A_node_x[condition_x]
    normals_y = grad_A_node_y[condition_y]

    new = A_node*(1.0-A_node)

    #new = np.sqrt(grad_A_node_x**2+grad_A_node_y**2)  
    #new[indexes_x] = normals_x
    #new[indexes_y] = normals_y

    print(np.max(grad_A_node_y))
    print(np.max(grad_A_node_x))
    print(np.max(new))
    print(np.max(A_node))

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("science")
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rcParams.update({"font.size": 28})


    fig, ax = plt.subplots(figsize=(16,8))
    
    im = ax.imshow(np.real(np.gradient(A)[0]*(1-A)), aspect='auto', cmap='inferno', interpolation='none', origin='lower')
    ax.set_xlabel('$X$ (nm)')
    ax.set_ylabel('$Y$ (nm)')

    plt.show()

    raise()

    return indexes_x, indexes_y, normals_x, normals_y

def find_boundaries_projection(dis, A, edofMat, nElx, nEly):

    """
    Finds the indexes of the boundary elements.
    @ A: Material interpolation
    """

    sigma_x = np.array([-1,1,-1,1])
    sigma_y = np.array([1,1,-1,-1])

    P_x = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    P_y = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    dP_x = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    dP_y = np.zeros((nEly+1)* (nElx+1), dtype="complex128")

    rho_flat = A.flatten()

    for i, nodes in enumerate(edofMat):

        rho_e = rho_flat[i]

        P_x [nodes.astype(int)] += 0.5*(rho_e - 0.5)*sigma_x
        P_y [nodes.astype(int)] += 0.5*(rho_e - 0.5)*sigma_y
        dP_x [nodes.astype(int)] += 0.5*sigma_x
        dP_y [nodes.astype(int)] += 0.5*sigma_y

    P_x[dis.n2BC] = 0.0
    P_x[dis.n3BC] = 0.0
    P_y[dis.n1BC] = 0.0
    P_y[dis.n4BC] = 0.0
    dP_x[dis.n2BC] = 0.0
    dP_x[dis.n3BC] = 0.0
    dP_y[dis.n1BC] = 0.0
    dP_y[dis.n4BC] = 0.0

    P_x = np.reshape(P_x, (nEly+1, nElx+1))
    P_y = np.reshape(P_y, (nEly+1, nElx+1))
    dP_x = np.reshape(dP_x, (nEly+1, nElx+1))
    dP_y = np.reshape(dP_y, (nEly+1, nElx+1))

    indexes_x = np.where(P_x != 0.0)
    indexes_y = np.where(P_y != 0.0)

    normals_x = P_x[indexes_x]
    normals_y = P_y[indexes_y]
    dnormals_x = dP_x[indexes_x]
    dnormals_y = dP_y[indexes_y]

    #plt.rcParams.update(plt.rcParamsDefault)
    #plt.style.use("science")
    #import matplotlib as mpl
    #from mpl_toolkits.axes_grid1 import make_axes_locatable

    #mpl.rcParams.update({"font.size": 28})


    #fig, ax = plt.subplots(figsize=(16,8))
    
    #im = ax.imshow(np.real(P_x), aspect='auto', cmap='inferno', interpolation='none', origin='lower')
    #ax.quiver(0.25*np.real(P_x), -0.25*np.real(P_y), scale=15)
    #ax.set_xlabel('$X$ (nm)')
    #ax.set_ylabel('$Y$ (nm)')

    #plt.show()

    #raise()

    return indexes_x, indexes_y, normals_x, normals_y, dnormals_x, dnormals_y
