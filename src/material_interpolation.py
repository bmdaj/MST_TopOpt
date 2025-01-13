import numpy as np

def material_interpolation_sc(eps_r, x, alpha_i=0.0):
    """
    Function that implements the material interpolation for a semiconductor.
    It returns the interpolated field and its derivative with respect to the position.
    The non-physical imaginary part discourages intermediate values of the design variable.
    @ eps_r: relative permittivity of the semiconductor.
    @ x: position of the design variable.
    @ alpha_i: problem dependent sacling factor for the imaginary term.
    """

    A = 1 + x*(eps_r-1) - 1j * alpha_i * x * (1 - x)
    dAdx = (eps_r-1) - 1j * alpha_i * (1 - 2*x)

    return A, dAdx

def material_interpolation_metal(n_metal, k_r, x, alpha_i=0.0):
    """
    Function that implements the material interpolation for a metal.
    It returns the interpolated field and its derivative with respect to the position.
    It avoids artificial resonances in connection with transition from positive to negative dielectric index.
    @ n_r: refractive index of the metal.
    @ k_r: extinction cross section of the metal.
    @ x: position of the design variable
    @ alpha_i: problem dependent sacling factor for the imaginary term.
    """

    n_eff = 1.0 + x*(n_metal-1.0) # effective refractive index
    k_eff = 0 + x*(k_r-0) # effective wavevector

    
    A = (n_eff**2-k_eff**2)-1j*(2*n_eff*k_eff) - 1j* alpha_i * x*(1-x)
    dAdx = 2*n_eff*(n_metal-1.0)-2*k_eff*(k_r-0)-1j*(2*(n_metal-1.0)*k_eff+2*n_eff*(k_r-0)) - 1j* alpha_i * (1 - 2* x)

    return A, dAdx

def material_interpolation_heat(k_mat, k_back, x):
    """
    Function that implements the material interpolation for the heat problem.
    It returns the interpolated field and its derivative with respect to the position.
    @ k_metal: conduction coefficient for the design material.
    @ k_back: conduction coefficient for the backround medium.
    """

    k_bck = k_back * np.ones_like(x, dtype="complex128")

    A = k_bck + x*(k_mat-k_bck) 
    dAdx = k_mat-k_bck

    return A, dAdx