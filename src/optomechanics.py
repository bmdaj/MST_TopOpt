import numpy as np

def calc_MST(Ex,Ey,Ez,Hx,Hy,Hz):

    """
    Calculates the time-averaged Maxwell Stress Tensor (MST) for an electromagnetic field.
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
    Calculates derivative with respect to the field solution (E_z) of the time-averaged Maxwell Stress Tensor (MST) for an electromagnetic field.
    The input parameters (e.g. alphaz, betax) are derived in the manuscript to build up the derivative tensor.
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
    

def find_boundaries_projection(dis, A, edofMat, nElx, nEly):

    """
    Finds the indexes of nodes at the boundary and the normals for each one of the nodes.
    @ A: Material interpolation matrix.
    @ edofmat: Matrix with the degrees-of-freedom per element in the finite-element discretization.
    @ nElx: Number of elements in the X direction.
    @ nEly: Number of elements in the Y direction.
    """

    sigma_x = np.array([-1,1,-1,1]) # matrix to be used in a later operation
    sigma_y = np.array([1,1,-1,-1]) # matrix to be used in a later operation

    P_x = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    P_y = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    dP_x = np.zeros((nEly+1)* (nElx+1), dtype="complex128")
    dP_y = np.zeros((nEly+1)* (nElx+1), dtype="complex128")

    rho_flat = A.flatten()

    for i, nodes in enumerate(edofMat):

        rho_e = rho_flat[i]

        P_x [nodes.astype(int)] += 0.5*(rho_e - 0.5)*sigma_x # this operation helps to identify which nodes are at a boundary with respect to the X axis.
        P_y [nodes.astype(int)] += 0.5*(rho_e - 0.5)*sigma_y # this operation helps to identify which nodes are at a boundary with respect to the Y axis.
        dP_x [nodes.astype(int)] += 0.5*sigma_x # derivative with respect to the design variables
        dP_y [nodes.astype(int)] += 0.5*sigma_y # derivative with respect to the design variables

    # If nodes are at the ends of the simulation domain , don't count them as being at the boundary of the particle

    P_x[dis.n2BC] = 0.0
    P_x[dis.n3BC] = 0.0
    P_y[dis.n1BC] = 0.0
    P_y[dis.n4BC] = 0.0
    dP_x[dis.n2BC] = 0.0
    dP_x[dis.n3BC] = 0.0
    dP_y[dis.n1BC] = 0.0
    dP_y[dis.n4BC] = 0.0

    # We reshape the matrices

    P_x = np.reshape(P_x, (nEly+1, nElx+1))
    P_y = np.reshape(P_y, (nEly+1, nElx+1))
    dP_x = np.reshape(dP_x, (nEly+1, nElx+1))
    dP_y = np.reshape(dP_y, (nEly+1, nElx+1))

    # We find the indexes of the nodes by identifying the nonzero matrix entries

    indexes_x = np.where(P_x != 0.0)
    indexes_y = np.where(P_y != 0.0)

    # By using the indexes we find the normals and their derivatives for the X and Y directions

    normals_x = P_x[indexes_x]
    normals_y = P_y[indexes_y]
    dnormals_x = dP_x[indexes_x]
    dnormals_y = dP_y[indexes_y]

    return indexes_x, indexes_y, normals_x, normals_y, dnormals_x, dnormals_y
