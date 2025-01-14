import numpy as np

class phy:
    " Class that defines the physics of the model."

    def __init__(self,
                 eps,
                 part_shape,
                 part_size,
                 part_center,
                 eps_part,
                 scale,
                 wavelength,
                 alpha):
        """
        @ eps: Value for the material's dielectric constant.
        @ part_shape: parameter that controls the shape of the particle: i.e. "circle" or "square".
        @ part_size: parameter that controls the size of the particle.
        @ part_center: center of the particle.
        @ eps_part: parameter that controls the dielectric constant of the particle.
        @ scale: scale of the physical problem; i.e. 1e-9 for nm.
        @ wavelength: Wavelength of the problem (Frequency domain solver).
        @ alpha: The attenuation factor (optical losses) added to our materials.
        """

        self.eps = eps
        self.part_size = part_size
        self.part_shape = part_shape
        self.part_center = part_center
        self.eps_part = eps_part
        self.scale = scale
        self.wavelength = wavelength
        self.k = 2 * np.pi / (self.wavelength * self.scale)
        c = 299792458 # speed of light
        self.mu_0 = 1.2566370616244444E-6 # vacuum magnetic constant
        self.eps_0 = 8.854187816292039E-12 # vacuum electric constant
        self.omega = self.k * c
        self.alpha = alpha


