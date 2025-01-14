import numpy as np
from .element_matrices import field_gradient, field_gradient_cross
from .optomechanics import calc_dMSTdEz

def calc_AdjRHS(dis, phy):

    def sum_nodes_x(x):
            """
            Function that is equivalent to the derivative of the integral over the nodes at the boundary of the particle.
            This function targets the nodes with a normal component in the X direction.
            @ x: field to be derivated over the sum of the nodes.
            """
            y = np.zeros_like(x).flatten()
            y[dis.nodes_b_x] = x[dis.nodes_b_x]*dis.b_n_x *dis.scaling
            return y

    def sum_nodes_y(x):
            """
            Function that is equivalent to the derivative of the integral over the nodes at the boundary of the particle.
            This function targets the nodes with a normal component in the Y direction.
            @ x: field to be derivated over the sum of the nodes.
            """
            y = np.zeros_like(x).flatten()
            y[dis.nodes_b_y] = x[dis.nodes_b_y]*dis.b_n_y *dis.scaling
            return y

    def calc_terms(function):
        """
        Calculate the individual terms in the sensitivity of the MST, i.e. the derivative dMST/dEz.
        @ function: spatial operation. In this case, we need to use the derivative of the summation over the 
                    nodes in the boundary with the correct scaling and normal vector component.
        """

        # -------------------------------------------------------------------------------
        # Calculation of betayx and betaxy
        # -------------------------------------------------------------------------------

        inp1 =  function(np.conj(dis.Hy).flatten())
        inp2 =  function(np.conj(dis.Hx).flatten())

        coef  = field_gradient_cross(dis.edofMat, -inp2.flatten(), inp1.flatten() , dis.scaling)

        betayx =  -1j*coef.flatten() /(phy.omega*phy.mu_0)

        inp1 =  function(np.conj(dis.Hy).flatten())
        inp2 =  function(np.conj(dis.Hx).flatten())

        coef  = field_gradient_cross(dis.edofMat, -inp2.flatten(), inp1.flatten() , dis.scaling)

        betaxy =  -1j*coef.flatten() /(phy.omega*phy.mu_0)

        # -------------------------------------------------------------------------------
        # Calculation of betay and betax
        # ------------------------------------------------------------------------------

        inp = function(np.imag(dis.Hx).flatten())
        inp2 = function(np.real(dis.Hx).flatten())
        _, coef  = field_gradient(dis.edofMat, inp.flatten(), dis.scaling)
        _, coef1  = field_gradient(dis.edofMat, inp2.flatten(), dis.scaling)
        betax =  -2*(coef+1j*coef1).flatten() /(phy.omega*phy.mu_0)

        inp = function(np.conj(dis.Hy).flatten())
        coef, _  = field_gradient(dis.edofMat, inp.flatten(), dis.scaling)
        betay =  2*1j*(coef).flatten() /(phy.omega*phy.mu_0)

        # -------------------------------------------------------------------------------
        # Calculation of alphaz
        # ------------------------------------------------------------------------------

        alphaz = function(((2 * np.real(dis.Ez) - 1j* 2*np.imag(dis.Ez))).flatten())

        return betayx, betaxy, betax, betay, alphaz
    
    # We calculate the individual parameters in the MST sensitivity.

    betayx_x, betaxy_x, betax_x, betay_x, alphaz_x = calc_terms(sum_nodes_x)
    betayx_y, betaxy_y, betax_y, betay_y, alphaz_y = calc_terms(sum_nodes_y)

    # We calculate the derivative of the MST components with respect to field solution (E_z).


    _, dT_pxdEz_x, dT_pydEz_x = calc_dMSTdEz(alphaz_x, betax_x, betay_x, betaxy_x, betayx_x)
    _, dT_pxdEz_y, dT_pydEz_y = calc_dMSTdEz(alphaz_y, betax_y, betay_y, betaxy_y, betayx_y)

    # Now we calculate the RHS by using both contributions of the projection from the Maxwell Stress Tensor.

    AdjRHS = (dT_pydEz_x[0,:]+dT_pydEz_y[1,:])

    return AdjRHS
    