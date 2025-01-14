from .discretization import dis
from .heat_discretization import dis_heat
import numpy as np 
from .physics import phy
from .filter_threshold import filter_threshold
from .plot import *
import warnings
import time 
from .optimization import optimizer
from .logfile import create_logfile_optimization, init_dir
from .plot import plot_it_history

class TopOpt_solver:
    """
    Main class for the 2D Topology Optimization framework in 2D.
    It may be used to:
    a) Run the forward problem for a given configuration of the dielectric function.
    b) Run an inverse design problem using the Topology Optimization framework. 
    """
    def __init__(self,
                 center_part,
                 dVElmIdx,
                 dVElmIdx_part,
                 dVElmIdx_part_pad,
                 nElX, 
                 nElY,
                 DVini,
                 DVini_part,
                 eps, 
                 wl,  
                 fR, 
                 eta,
                 beta, 
                 scaling, 
                 part_shape,
                 part_size,
                 eps_part):
        """
        Initialization of the main class.
        @ center_part: Geometric center of the particle
        @ dVElmIdx : Indexes for the design variables for the metalens.
        @ dVElmIdx_part : Indexes for the design variables for the particle.
        @ dVElmIdx_part_pad : Indexes for a padded version of the design variables for the particle.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ dVini: Initial value for the design variables for the metalens.
        @ dVini_part: Initial value for the design variables for the particle.
        @ eps: Value for the material's dielectric constant.
        @ wl: Wavelength of the problem (Frequency domain solver).
        @ fR: Filtering radius.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ eta: parameter that controls threshold value.
        @ beta: parameter that controls the threshold sharpness.
        @ scaling: Scale of the physical problem: length of the side of the finite element.
        @ part_shape: parameter that controls the shape of the particle: i.e. "circle" or "square".
        @ part_size: parameter that controls the size of the particle.
        @ eps_part: parameter that controls the dielectric constant of the particle.
        """
        warnings.filterwarnings("ignore") # we eliminate all possible warnings to make the notebooks more readable.
        self.center_part = center_part
        self.dVElmIdx = dVElmIdx
        self.dVElmIdx_part = dVElmIdx_part
        self.dVElmIdx_part_pad = dVElmIdx_part_pad
        
        self.nElX = nElX
        self.nElY = nElY
        self.eps = eps
        self.wavelength = wl
        self.fR = fR

        self.dVini = DVini
        self.dVini_part = DVini_part
        self.dis_0 = None
        self.dVs = None
        self.eta = eta
        self.beta = beta
        self.eta_con = 0.5 # threshold value fot the connectivity (heat) problem

        self.part_shape = part_shape
        self.part_size = part_size
        self.eps_part = eps_part
        self.alpha = 0.0 # parameter to introduce artificial attenuation in the material interpolation


        # -----------------------------------------------------------------------------------
        # PHYSICS OF THE PROBLEM
        # ----------------------------------------------------------------------------------- 
        self.scaling = scaling # We give the scaling of the physical problem; i.e. 20e-9 for 20 nm.

        self.phys = phy(self.eps,
                        self.part_shape,
                        self.part_size,
                        self.center_part, 
                        self.eps_part,
                        self.scaling,
                        self.wavelength,
                        self.alpha) 

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE PROBLEM
        # ----------------------------------------------------------------------------------- 
                        
        center_part_idx = (self.center_part[1]-1)*self.nElX+self.center_part[0]-1                
        self.dis_0 = dis(self.scaling,
                    self.nElX,
                    self.nElY,
                    center_part_idx,
                    self.dVElmIdx,
                    self.dVElmIdx_part,
                    self.dVElmIdx_part_pad)

        # We set the indexes of the discretization: i.e. system matrix, boundary conditions ...

        self.dis_0.index_set() 

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE HEAT PROBLEM FOR THE CONNECTIVITY CONSTRAINT
        # -----------------------------------------------------------------------------------    

        self.nEly_lens = len(self.dVElmIdx[0]) #  number of elements in the Y axis of the metalens region
        self.nElx_lens = len(self.dVElmIdx[1]) #  number of elements in the X axis of the metalens region

        self.nEly_part = len(self.dVElmIdx_part[0]) #  number of elements in the Y axis of the particle region
        self.nElx_part = len(self.dVElmIdx_part[1]) #  number of elements in the X axis of the particle region


        self.dis_heat = dis_heat(self.scaling,
                    self.nElx_lens,
                    self.nEly_lens)

        self.dis_heat_part = dis_heat(self.scaling,
                    self.nElx_part,
                    self.nEly_part)

        # We set the indexes of the discretization for the heat problem: i.e. system matrix, boundary conditions ...

        self.dis_heat.index_set(obj="lens") 
        self.dis_heat_part.index_set(obj="part") 

        # -----------------------------------------------------------------------------------  
        # FILTERING AND THRESHOLDING 
        # -----------------------------------------------------------------------------------  
        #  
        self.filThr =  filter_threshold(self.fR, self.nElX, self.nElY, self.eta, self.beta) 
        self.filThr_con =  filter_threshold(self.fR, self.nElx_lens, self.nEly_lens, self.eta_con, self.beta) 
        self.filThr_con_part =  filter_threshold(self.fR, self.nElx_part, self.nEly_part, self.eta_con, self.beta)

        # -----------------------------------------------------------------------------------  
        # INITIALIZING DESIGN VARIABLES
        # -----------------------------------------------------------------------------------  
        
        self.dVs = self.dVini 
        self.dVs_part = self.dVini_part 

        # -----------------------------------------------------------------------------------  
        # INITIALIZING LOGFILE
        # -----------------------------------------------------------------------------------  

        self.logfile = False

        if self.logfile:
            self.directory_opt, self.today = init_dir("_opt")
    

    def solve_forward(self, dVs, dVs_part):
        """
        Function to solve the forward FEM problem in the frequency domain given a distribution of dielectric function in the simulation domain.
        """
        self.Ez = self.dis_0.FEM_sol(dVs, dVs_part, self.phys, self.filThr)
        self.FOM =  self.dis_0.compute_FOM()
        _, self.sens, self.sens_part = self.dis_0.objective_grad(dVs, dVs_part, self.phys, self.filThr)

        return self.Ez, self.FOM


    
    def optimize(self, maxItr):
        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ maxItr: Maximum number of iterations of the optimizer. 
        """
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs_part)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs_part)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION AND CONSTRAINTS USED BY THE OPTIMIZER
        # ----------------------------------------------------------------------------------- 

        global i_con # We define an iteration number for the continuation steps
        global it_num # We define a global number of iterations to keep track of the step
        global iteration_number_list
        it_num = 0
        i_con = 0
        self.maxItr = maxItr
        it_num = self.maxItr
        iteration_number_list = []
        self.continuation_scheme = True # Flag to enable or disable the continuation scheme. It would be nice to add as an argument.

        def f0(x, it_num):
            """
            Main objective function to be optimized. In this case it solves the finite-element problem and outputs the FOM and the sensitivities.
            @ x: design variables.
            @ it_num: iteration number to be followed for the continuation process.
            """
            global i_con

            dVs = np.zeros_like(self.dVs)
            dVs_part = x
            FOM_old = self.FOM
            self.FOM, sens, sens_part = self.dis_0.objective_grad(dVs, dVs_part, self.phys, self.filThr) # solves the PDE and outputs FOM and sensitivities

            if self.logfile:
                save_designs(self.nElX, self.nElY, self.scaling, self.dis_0, it_num, self.directory_opt)
                self.FOM_list[it_num] = self.FOM
                self.iteration_history(it_num, save=True, dir=self.directory_opt)
            
            if self.continuation_scheme:
                if it_num>0 and np.abs(FOM_old-self.FOM)<5E-4: #convergence criterion for the continuation
                    if self.beta<75.0:
                        self.beta =  self.beta*1.5
                        self.alpha += 0.1
                    else:
                        self.beta = self.beta
                        self.alpha += 0.1
                        
                    print("NEW BETA: ", self.beta)
                    print("NEW ALPHA: ", self.alpha)

                    # At every new continuation step we need to re-set the filter and physics with the new beta and alpha values

                    self.filThr =  filter_threshold(self.fR, self.nElX, self.nElY, self.eta, self.beta) 
                    self.filThr_con =  filter_threshold(self.fR, self.nElx_lens, self.nEly_lens, self.eta_con, self.beta) 
                    self.filThr_con_part =  filter_threshold(self.fR, self.nElx_part, self.nEly_part, self.eta_con, self.beta)
                    self.phys = phy(self.eps,
                            self.part_shape,
                            self.part_size,
                            self.center_part, 
                            self.eps_part,
                            self.scaling,
                            self.wavelength,
                            self.alpha) 

                    i_con += 1
                    iteration_number_list.append(it_num+1)

            it_num += 1
            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)

            
            return self.FOM, sens_part.flatten()[:, np.newaxis]


        def con_part(x):
            """
            Constraint for the particle connectivity formulated as an artificial heat problem.
            @ x: design variables.
            """

            self.dVs_part = x
            self.part_domain = np.reshape(self.dVs_part, (self.nEly_part, self.nElx_part))
            FOM , self.sens_heat_part = self.dis_heat_part.objective_grad(self.part_domain, self.filThr_con_part) # solve PDE and calculate sensitivities.

            con_const =  ((FOM-self.eps_con_part)/self.eps_con_part).astype("float64")
            
            print("Connectivity constraint:", con_const)
            
            return  con_const , self.sens_heat_part.flatten()/self.eps_con_part

        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------

        n = len(self.dVs_part) # number of parameters to optimize
        self.FOM_list = np.zeros(maxItr)

        # -----------------------------------------------------------------------------------  
        # INITIALIZE OPTIMIZER
        # -----------------------------------------------------------------------------------

        m = 2 # number of constraint: 2 objective functions in minmax, 1 volume constraint, 2 geometric lengthscale constraint
        p = 0 # # number of objective functions in minmax
        f = np.array([con_part]) # constraint list
    
        a0 = 1.0 
        a = np.zeros(m)[:,np.newaxis] 
        d = np.zeros(m)[:,np.newaxis]
        c = 1000 * np.ones(m)[:,np.newaxis]
        move = 0.1 # 0.2 for easy, 0.1 for hard problems.
        self.eps_con = 0.99 # might need tweaking if optimization problem changes, usually good to normalize all FOMs and constraints to 1
        self.eps_con_part = 0.90 # might need tweaking if optimization problem changes, usually good to normalize all FOMs and constraints to 1
        
        self.opt = optimizer(m, n, p, LBdVs[:,np.newaxis], UBdVs[:,np.newaxis], f0, f, a0, a, c, d, self.maxItr, move, type_MMA="MMA")

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme: # if the continuation scheme is active, how to change the beta values     
            factor = 1.5
            betas = self.beta * np.array([factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7, factor**8, factor**9, factor**10])
            

        start = time.time() # we track the total optimization time

        self.dVs_part, self.FOM_list, _, = self.opt.optimize(self.dVs_part[:,np.newaxis])
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")

        if self.logfile:
            create_logfile_optimization(self)

        return self.dVs_part
    

    def calculate_forces(self):
        """
        Gives the value of the forces on the particle in pN/μm.
        """
        Fx = np.real(self.dis_0.Fx)*1E6 # multiply to set correct units
        Fy = np.real(self.dis_0.Fy)*1E6 # multiply to set correct units
        print("Fx (pN/μm): ", Fx)
        print("Fy (pN/μm): ", Fy)

    def plot_H_field(self, comp):
        """
        Plots the magnetic field components.
        @ comp: Component of the magnetic field, "x" or "y" for TE polatization.
        """
        plot_H_comp(self.dis_0, comp)

    def plot_E_field(self):
        """
        Plots the electric field component, only "z" for TE polarization.
        """
        plot_E_comp(self.dis_0, self.Ez)


    def iteration_history(self):
        """
        Plots the iteration history, with the FOM and the connectivity constraint.
        """

        print("----------------------------------------------")
        print("Iteration history")
        print("----------------------------------------------")

        plot_it_history(self.FOM_list, self.opt.cons_it, it_num)
        
