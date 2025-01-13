import numpy as np

def ip_quad_1o(a,b): 
    """
    Implementation for the interpolation functions of first order for linear rectangular elements.
    In our case we assume linear rectangular elements with local indexing:
    1 2 
    3 4.
    @ a: side of the rectangle in the X direction.
    @ b: side of the rectangle in the Y direction.
    """

    surface = 4* a * b

    N_3 = lambda x,y:   (a-x) * (b-y) / surface
    N_4 = lambda x,y:  (a+x) * (b-y) / surface
    N_1 = lambda x,y: (a-x) * (b+y) / surface
    N_2 = lambda x,y:  (a+x) * (b+y) / surface

    return N_1, N_2, N_3, N_4

def ip_line_1o(): 
    """
    Implementation for the interpolation functions of first order for linear line elements.
    In our case we assume linear line elements with local indexing:
    1 2 
    """

    N_1 = lambda x: 1-x
    N_2 = lambda x: x

    return N_1, N_2

def ip_quad_1o_dx(a,b): 
    """
    Implementation for the derivative of the interpolation functions of first order with respect to X for linear rectangular elements.
    In our case we assume linear rectangular elements with local indexing:
    1 2 
    3 4.
    @ a: side of the rectangle in the X direction.
    @ b: side of the rectangle in the Y direction.
    """

    surface = 4* a * b

    N_3 = lambda y:  - (b-y) / surface
    N_4 = lambda y:   (b-y) / surface
    N_1 = lambda y: - (b+y) / surface
    N_2 = lambda y:  (b+y) / surface

    return N_1, N_2, N_3, N_4

    
def ip_quad_1o_dy(a,b): 
    """
    Implementation for the derivative of the interpolation functions of first order with respect to Y for linear rectangular elements.
    In our case we assume linear rectangular elements with local indexing:
    1 2 
    3 4.
    @ a: side of the rectangle in the X direction.
    @ b: side of the rectangle in the Y direction.
    """

    surface = 4* a * b

    N_3 = lambda x:   -(a-x) / surface
    N_4 = lambda x:  -(a+x) / surface
    N_1 = lambda x: (a-x) / surface
    N_2 = lambda x:  (a+x) / surface

    return N_1, N_2, N_3, N_4

    
def calculate_integral(f, a, b):
    """
    Function to calculate surface integrals in two dimensional spaces.
    @
    @ a: side of the rectangle in the X direction (i.e. start/end of integration domain).
    @ b: side of the rectangle in the Y direction (i.e. start/end of integration domain).
    """

    import scipy.integrate as integrate
    return integrate.dblquad(f, -a, a, -b, b)

def calculate_integral_1D(f, a, b):
    """
    Function to calculate the definite integral of a one-dimensional function.
    
    @param f: The function to integrate.
    @param a: The start of the integration interval.
    @param b: The end of the integration interval.
    
    @return: The value of the definite integral of f from a to b.
    """
    import scipy.integrate as integrate
    return integrate.quad(f, a, b)[0]


def calc_d(field, ipf_dy, x):
    """"
    Calculates the summed derivative of a field using the interpolation function.
    
    @field : The function to integrate.
    @ipf_dy: Derivated interpolation function (i.e. df/dy)
    @x: dependent variable.
    """

    total = 0.0 + 0.0j

    for i in range(len(ipf_dy)):

        total += field[i] * ipf_dy[i](x)

    return total

def element_matrices(scaling):
    """
    Calculates the element matrices used to construct the global system matrix.
    In our case we assume linear rectangular elements with local indexing:
    1 2 
    3 4.
    """

    a = scaling / 2 # Element size scaling
    b = scaling / 2 # Element size scaling


    # -----------------------------------------------------------------------------------
    # CALCULATION OF THE MASS MATRIX
    # ----------------------------------------------------------------------------------- 

    ipf = list(ip_quad_1o(a,b)) # initialize the interpolation functions
    MEM = np.zeros((4,4))

    # We fill the matrix components for the Mass matrix using the interpolation functions

    for i in range(4):
        for j in range(4):
            integrand = lambda x, y : ipf[i] (x, y) * ipf[j] (x, y) 
            MEM [i,j] = calculate_integral(integrand, a, b) [0]

    # -----------------------------------------------------------------------------------
    # CALCULATION OF THE LAPLACE MATRIX
    # ----------------------------------------------------------------------------------- 

    ipf = list(ip_quad_1o(a,b)) # initialize the interpolation functions
    ipf_dx = list(ip_quad_1o_dx(a,b)) # initialize the dX of interpolation functions
    ipf_dy = list(ip_quad_1o_dy(a,b)) # initialize the dY of interpolation functions

    LEM = np.zeros((4,4))

    # We fill the matrix components for the Laplace matrix using the interpolation functions

    for i in range(4):
        for j in range(4):

            integrand = lambda x, y : ipf_dx [i] (y) * ipf_dx [j] (y) +  ipf_dy [i] (x) * ipf_dy [j] (x)
            LEM [i,j] = calculate_integral(integrand, a, b) [0]

    return LEM, MEM

def boundary_element_matrix (scaling, k, eps):
    """
    Calculates the boundary element matrices used to define the boundary conditions of the system.
    In our case we assume linear rectangular elements and Neumann type boundary conditions.
    @ scaling: scaling of the physical problem; i.e. 1e-9 for nm.
    @ k: wavevector of the physical problem (frequency domain solver).
    @ eps: permittivity of the boundary material.
    """

    #KSM = 1j * k* scaling * np.array([[1/3, 1/6],
    #                                  [1/6, 1/3]])

    ipf = list(ip_line_1o()) # initialize the interpolation functions

    gamma = 1j * k * scaling *np.sqrt(eps)

    KSM = np.zeros((2,2))

    for i in range(2):
        for j in range(2):

            integrand = lambda x : ipf [i] (x) * ipf [j] (x)
            KSM [i,j] = calculate_integral_1D(integrand, 0.0, 1.0)                        

    return gamma * KSM

def field_gradient(edofMat, field, scaling):
    """
    Calculates the gradient of a field in the physical system.
    @ edofMat: matrix with the degree of freedom numbering on an element basis.
    @ field: The field to be differentiated.
    @ scaling: scaling of the physical problem; i.e. 1e-9 for nm.
    """

    a = scaling / 2 # Element size scaling
    b = scaling / 2 # Element size scaling


    node1_x = -a
    node1_y = b

    node2_x = a
    node2_y = b

    node3_x = -a
    node3_y = -b

    node4_x = a
    node4_y = -b

    ipf_dx = list(ip_quad_1o_dx(a, b))
    ipf_dy = list(ip_quad_1o_dy(a, b))
    ipf = list(ip_quad_1o(a, b))


    f_x = np.zeros_like(field, dtype="complex128")
    f_y = np.zeros_like(field, dtype="complex128")

    for nodes in edofMat:

        n = nodes.astype(int)

        field_nodes = field[n].flatten()

        f_y_elem = lambda x : calc_d(field_nodes, ipf_dy, x)
        f_y[n] += 0.25 * np.array([f_y_elem(node1_x), f_y_elem(node2_x), f_y_elem(node3_x), f_y_elem(node4_x)])

        f_x_elem = lambda y : calc_d(field_nodes, ipf_dx, y)
        f_x[n] += 0.25 * np.array([f_x_elem(node1_y), f_x_elem(node2_y), f_x_elem(node3_y), f_x_elem(node4_y)])


    return f_x, f_y

def field_gradient_cross(edofMat, field1, field2, scaling):
    """
    Calculates the sum of the spatial derivative for two dimensions for a field in the physical system.
    @ edofMat: matrix with the degree of freedom numbering on an element basis.
    @ field: The field to be differentiated.
    @ scaling: scaling of the physical problem; i.e. 1e-9 for nm.
    """

    a = scaling / 2 # Element size scaling
    b = scaling / 2 # Element size scaling

    node1_x = -a
    node1_y = b

    node2_x = a
    node2_y = b

    node3_x = -a
    node3_y = -b

    node4_x = a
    node4_y = -b

    ipf_dx = list(ip_quad_1o_dx(a, b))
    ipf_dy = list(ip_quad_1o_dy(a, b))

    f_xy = np.zeros_like(field1, dtype="complex128")

    for nodes in edofMat:

        n = nodes.astype(int)

        field_nodes1 = field1[n].flatten()
        field_nodes2 = field2[n].flatten()

        f_x_elem = lambda y : calc_d(field_nodes1, ipf_dx, y)
        f_y_elem = lambda x : calc_d(field_nodes2, ipf_dy, x)
        
        f_xy[n] += 0.25 * np.array([f_x_elem(node1_y)+f_y_elem(node1_x), f_x_elem(node2_y)+f_y_elem(node2_x), f_x_elem(node3_y)+f_y_elem(node3_x), f_x_elem(node4_y)+f_y_elem(node4_x)])

    return f_xy