# `MST_TopOpt`

This repository contains the code used to reproduce key results from the paper:

**B. Martinez de Aguirre Jokisch, R.E. Christiansen, O. Sigmund.**  
*"Engineering optical forces through Maxwell stress tensor inverse design"*  
arXiv preprint [arXiv:2410.20009](https://arxiv.org/abs/2410.20009) (2024).

## Overview

Precise spatial manipulation of particles via optical forces is essential in many research areas, ranging from biophysics to atomic physics. Central to this effort is the challenge of designing optical systems optimized for specific applications. Traditional design methods often rely on trial-and-error approaches or simplified models, such as approximating particles as point dipoles—an assumption valid only for particles much smaller than the wavelength of the electromagnetic field.

In this work, we present a general **inverse design framework** based on the **Maxwell stress tensor (MST) formalism**. This framework enables the simultaneous design of all components of the system and is applicable to particles of arbitrary sizes and shapes. With small modifications to the baseline formulation, the method can engineer systems capable of attracting, repelling, accelerating, oscillating, and trapping particles.


The methodology relies on the **finite element method (FEM)** and **topology optimization**, a gradient-based approach for iteratively designing optical systems. The examples in this work are **two-dimensional**, assuming **transverse electric (TE) polarization**, with the optical system illuminated by an **incident plane wave**. Note that the base-code can be modified to assume transverse magnetic (TM) polarization or be illuminated by more complex sources.

### Installation

To install the `MST_TopOpt` package, use the following command:

```bash
pip install MST_TopOpt
```

### Repository Content

This repository includes tutorials to reproduce the following results from the paper:

1. **Force Calculations for a square article in free-space:** The tutorial `square.ipynb` reproduces the results shown in Figure 3.  
2. **Optimization for a particle in free-space:** The tutorial `opt_repulsive.ipynb` implements the optimization process for free-space particles in Figure 4.

The code in this repository can be extended to reproduce additional results presented in the paper, such as those for particle-metalens systems and their applications.

## Citing `MST_TopOpt`

If you use `MST_TopOpt` in your research, we kindly request that you cite the following paper:

**B. Martinez de Aguirre Jokisch, R.E. Christiansen, O. Sigmund.**  
*"Engineering optical forces through Maxwell stress tensor inverse design"*  
arXiv preprint [arXiv:2410.20009](https://arxiv.org/abs/2410.20009) (2024).

### Example Citation in BibTeX:

```bibtex
@article{martinez2024msttopopt,
  title = {Engineering optical forces through Maxwell stress tensor inverse design},
  author = {Martinez de Aguirre Jokisch, B. and Christiansen, R.E. and Sigmund, O.},
  journal = {arXiv preprint},
  year = {2024},
  eprint = {2410.20009},
  archivePrefix = {arXiv}
}
