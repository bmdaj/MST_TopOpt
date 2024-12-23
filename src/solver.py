from discretization import dis
from heat_discretization import dis_heat
import numpy as np 
from physics import phy
from filter_threshold import filter_threshold
from plot import plot_intensity, plot_mi, plot_mi_heat, plot_sens, plot_sens_part, save_designs, plot_H_comp, plot_heat, plot_sens_heat
import warnings
import nlopt
import time 
from optimization import optimizer
from logfile import create_logfile_optimization, init_dir
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot import plot_it_history

class freq_top_opt_2D:
    """
    Main class for the 2D Topology Optimization framework in 2D.
    It may be used to:
    a) Run the forward problem for a given configuration of the dielectric function.
    b) Run an inverse design problem using the Topology Optimization framework. 
    """
    def __init__(self,
                 targetXY,
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
        @ target XY: Target element for the intensity FOM optimization.
        @ dVElmIdx : Indexes for the design variables.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ dVini: Initial value for the design variables.
        @ eps: Value for the material's dielectric constant.
        @ wl: Wavelength of the problem (Frequency domain solver).
        @ fR: Filtering radius.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ eta: parameter that controls threshold value.
        @ scaling: Scale of the physical problem.
        @ beta: parameter that controls the threshold sharpness.
        @ part_shape: parameter that controls the shape of the particle: i.e. "circle" or "square".
        @ part_size: parameter that controls the size of the particle.
        @ eps_part: parameter that controls the dielectric constant of the particle.
        """
        warnings.filterwarnings("ignore") # we eliminate all possible warnings to make the notebooks more readable.
        self.targetXY = targetXY
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
        self.eta_con = 0.5

        self.part_shape = part_shape
        self.part_size = part_size
        self.eps_part = eps_part
        self.alpha = 0.0


        # -----------------------------------------------------------------------------------
        # PHYSICS OF THE PROBLEM
        # ----------------------------------------------------------------------------------- 
        self.scaling = scaling # We give the scaling of the physical problem; i.e. 1e-9 for nm.

        self.phys = phy(self.eps,
                        self.part_shape,
                        self.part_size,
                        self.targetXY, 
                        self.eps_part,
                        self.scaling,
                        self.wavelength,
                        self.alpha) 

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE PROBLEM
        # -----------------------------------------------------------------------------------                 
        tElmIdx = (self.targetXY[1]-1)*self.nElX+self.targetXY[0]-1                
        self.dis_0 = dis(self.scaling,
                    self.nElX,
                    self.nElY,
                    tElmIdx,
                    self.dVElmIdx,
                    self.dVElmIdx_part,
                    self.dVElmIdx_part_pad)

        # We set the indexes of the discretization: i.e. system matrix, boundary conditions ...

        self.dis_0.index_set() 

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE PROBLEM
        # -----------------------------------------------------------------------------------    

        self.dis_0.index_set() 

        self.nEly_lens = len(self.dVElmIdx[0])
        self.nElx_lens = len(self.dVElmIdx[1])

        self.nEly_part = len(self.dVElmIdx_part[0])
        self.nElx_part = len(self.dVElmIdx_part[1])


        self.dis_heat = dis_heat(self.scaling,
                    self.nElx_lens,
                    self.nEly_lens,
                    debug=False)

        self.dis_heat_part = dis_heat(self.scaling,
                    self.nElx_part,
                    self.nEly_part,
                    debug=False)

        # We set the indexes of the discretization: i.e. system matrix, boundary conditions ...

        self.dis_heat.index_set(obj="lens") 
        self.dis_heat_part.index_set(obj="part") 

        # -----------------------------------------------------------------------------------  
        # FILTERING AND THRESHOLDING 
        # -----------------------------------------------------------------------------------   
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
        Ez = self.dis_0.FEM_sol(dVs, dVs_part, self.phys, self.filThr)
        self.FOM =  self.dis_0.compute_FOM()
        _, self.sens, self.sens_part = self.dis_0.objective_grad(dVs, dVs_part, self.phys, self.filThr)
        self.plot_FOM()

        return Ez, self.FOM
    
    def solve_scattered_field(self, dVs, dVs_part):
        """
        Function to solve the scattered field problem formulation
        """
        # First solve for the incident field
        Ez_inc = self.dis_0.FEM_sol(dVs, np.zeros_like(dVs_part), self.phys, self.filThr)
        self.plot_FOM()

        Ez_tot = self.dis_0.FEM_sol(dVs, dVs_part, self.phys, self.filThr)
        Ez_scat = Ez_tot - Ez_inc

        self.dis_0.Ez = Ez_scat

        # implement calculating the rest of the fields from this solution!
        
        self.plot_FOM()

        return Ez_scat



    def solve_heat(self, dVs, obj):

        if obj == "lens":
        
            self.lens_domain = np.reshape(dVs, (self.nEly_lens, self.nElx_lens))
            T = self.dis_heat.FEM_sol_heat(self.lens_domain, self.filThr_con)
            FOM =  self.dis_heat.compute_FOM()
            _, self.sens_heat = self.dis_heat.objective_grad(self.lens_domain, self.filThr_con)
            self.plot_heat(obj)

        if obj == "part":
        
            self.part_domain = np.reshape(dVs, (self.nEly_part, self.nElx_part))
            T = self.dis_heat_part.FEM_sol_heat(self.part_domain, self.filThr_con_part)
            FOM =  self.dis_heat_part.compute_FOM()
            _, self.sens_heat_part = self.dis_heat_part.objective_grad(self.part_domain, self.filThr_con_part)
            self.plot_heat(obj)

        return T

    
    def optimize(self, maxItr, algorithm):
        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ algorithm: Algorithm to be used by the optimizer i.e. MMA, BFGS. 
        """

        def set_optimization_algorithm(algorithm, n):
            if algorithm == "MMA":
                opt = nlopt.opt(nlopt.LD_MMA, n)
            if algorithm == "BFGS":
                opt = nlopt.opt(nlopt.LD_LBFGS, n)
            return opt
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs_part)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs_part)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 

        global i_con
        global it_num # We define a global number of iterations to keep track of the step
        global iteration_number_list
        it_num = 0
        i_con = 0
        self.maxItr = maxItr
        it_num = self.maxItr
        iteration_number_list = []
        self.continuation_scheme = True

        def f0(x, it_num):
            global i_con

            dVs = np.zeros_like(self.dVs)
            dVs_part = x
            FOM_old = self.FOM
            self.FOM, sens, sens_part = self.dis_0.objective_grad(dVs, dVs_part, self.phys, self.filThr)

            if self.logfile:
                save_designs(self.nElX, self.nElY, self.scaling, self.dis_0, it_num, self.directory_opt)
                self.FOM_list[it_num] = self.FOM
                self.iteration_history(it_num, save=True, dir=self.directory_opt)
            
            if self.continuation_scheme:
                #if (it_num+1) % 20 == 0 and (it_num+1) not in  iteration_number_list:
                if it_num>0 and np.abs(FOM_old-self.FOM)<5E-4:
                #if (it_num+1) % 50 == 0:
                    #self.opt.move = self.opt.move/1.5
                    #print("NEW BETA: ", betas[i_con])
                    if self.beta<75.0:
                        self.beta =  self.beta*1.5
                        self.alpha += 0.1
                    else:
                        self.beta = self.beta
                        self.alpha += 0.1
                        
                    #self.alpha = 0.0
                    print("NEW BETA: ", self.beta)
                    print("NEW ALPHA: ", self.alpha)
                    self.filThr =  filter_threshold(self.fR, self.nElX, self.nElY, self.eta, self.beta) 
                    self.filThr_con =  filter_threshold(self.fR, self.nElx_lens, self.nEly_lens, self.eta_con, self.beta) 
                    self.filThr_con_part =  filter_threshold(self.fR, self.nElx_part, self.nEly_part, self.eta_con, self.beta)
                    self.phys = phy(self.eps,
                            self.part_shape,
                            self.part_size,
                            self.targetXY, 
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

        def con_lens(x):

            self.dVs = x[:len(self.dVs)]
            self.lens_domain = np.reshape(self.dVs, (self.nEly_lens, self.nElx_lens))
            FOM , self.sens_heat = self.dis_heat.objective_grad(self.lens_domain, self.filThr_con)
            sens = np.zeros(len(self.dVs_part)+len(self.dVs))
            sens[:len(self.dVs)] = self.sens_heat.flatten()

            #self.plot_heat("lens")
            print("Lens constraint:", ((FOM-self.eps_con)/self.eps_con).astype("float64"))

            #plot_sens_heat(self.dis_heat, self.sens_heat)

            return ((FOM-self.eps_con)/self.eps_con), sens/self.eps_con

        def con_part(x):

            self.dVs_part = x
            self.part_domain = np.reshape(self.dVs_part, (self.nEly_part, self.nElx_part))
            FOM , self.sens_heat_part = self.dis_heat_part.objective_grad(self.part_domain, self.filThr_con_part)
            #sens = np.zeros(len(self.dVs_part)+len(self.dVs))
            #sens[len(self.dVs):] = self.sens_heat_part.flatten()



            #self.plot_sensitivities("heat", obj="part")
            print("MAX SENS PART:", np.max(np.abs(self.sens_heat_part)))
            #print("FOM: ", FOM)
            print("Particle constraint:", ((FOM-self.eps_con_part)/self.eps_con_part).astype("float64"))

            #plot_mi_heat(self.dis_heat_part)

            return  ((FOM-self.eps_con_part)/self.eps_con_part).astype("float64"), self.sens_heat_part.flatten()/self.eps_con_part


        
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
        f = np.array([]) #np.array([con_part])
    
        #f = np.array([])
        a0 = 1.0 # for minmax formulation 
        a = np.zeros(m)[:,np.newaxis] # p objective funcions and m constraints
        #a [0,0] = 1.0 
        d = np.zeros(m)[:,np.newaxis]
        c = 1000 * np.ones(m)[:,np.newaxis]
        move = 0.4 # check this with Jonathan at some point: 0.2 for easy, 0.1 for hard problems.
        self.eps_con = 0.99 #1.6 1#6 1.1, 0.5
        self.eps_con_part = 1.4 #1.025 #1, 1.30#8 1.23, 0.5
        
        self.opt = optimizer(m, n, p, LBdVs[:,np.newaxis], UBdVs[:,np.newaxis], f0, f, a0, a, c, d, self.maxItr, move, type_MMA="MMA")

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:     
            factor = 1.5
            betas = self.beta * np.array([factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7, factor**8, factor**9, factor**10])
            #betas = self.beta * np.array([factor, factor, factor, factor, factor, factor, factor])
            #self.alpha=0.2
            alphas = np.zeros_like(betas)#0.2 * np.array([1, factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7, factor**8, factor**9, factor**10])
            

        start = time.time() # we track the total optimization time

        #self.len_dVs_part = len(self.dVs_part)
        #self.len_dVs = len(self.dVs)
        #self.dVs_tot = np.concatenate([self.dVs.flatten(), self.dVs_part.flatten()])

        self.dVs_part, self.FOM_list, _, _, _ = self.opt.optimize(self.dVs_part[:,np.newaxis])
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")

        if self.logfile:
            create_logfile_optimization(self)

        # -----------------------------------------------------------------------------------  
        # FINAL BINARIZED DESIGN EVALUATION
        # -----------------------------------------------------------------------------------

        #print("----------------------------------------------")
        #print("Final binarized design evaluation")
        #print("----------------------------------------------")

        #dVs = self.dVs_tot[:len(self.dVs)]
        #dVs_part = self.dVs_tot[len(self.dVs):]
        
        #self.filThr.beta = 1000 # we set a very high treshold to achieve binarization
        #FOM, self.sens, self.sens_part = self.dis_0.objective_grad(dVs, dVs_part, self.phys, self.filThr) # we compute the FOM and sensitivities

        return self.dVs_part

    def sens_check (self, dVs):
        """
        Sensitivity check using finite-differences
        """ 
        delta_dV = 1e-5
        sens = np.zeros(len(dVs))
        self.Ez = self.dis_0.FEM_sol(dVs, self.dVs_part, self.phys, self.filThr)
        FOM_0 = self.dis_0.compute_FOM()
        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            self.Ez = self.dis_0.FEM_sol(dVs_new, self.dVs_part, self.phys, self.filThr)
            FOM_new = self.dis_0.compute_FOM()
            sens [i] = (FOM_new - FOM_0) / delta_dV
            print(FOM_new)
            dVs_new [i] -= delta_dV

        plot_sens(self.dis_0, sens)
        return sens


    def sens_check_part (self, dVs):
        """
        Sensitivity check using finite-differences
        """ 
        delta_dV = 1e-5
        sens = np.zeros(len(dVs))
        self.Ez = self.dis_0.FEM_sol(self.dVs, dVs, self.phys, self.filThr)
        FOM_0 = self.dis_0.compute_FOM()
        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            self.Ez = self.dis_0.FEM_sol(self.dVs, dVs_new, self.phys, self.filThr)
            FOM_new = self.dis_0.compute_FOM()
            sens [i] = (FOM_new - FOM_0) / delta_dV
            print(FOM_new)
            dVs_new [i] -= delta_dV

        plot_sens_part(self.dis_0, sens)
        return sens

    def sens_check_heat (self, dVs, obj):
        """
        Sensitivity check using finite-differences
        """ 
        delta_dV = 1e-5

        if obj == "lens":

            lens_domain = np.reshape(self.dVs, (self.nEly_lens, self.nElx_lens))

            sens = np.zeros(len(lens_domain.flatten()))
            _ = self.dis_heat.FEM_sol_heat(lens_domain, self.filThr_con)
            FOM_0 = self.dis_heat.compute_FOM()
            dVs_new = self.dVs

            for i in range(len(dVs)):
                dVs_new [i] += delta_dV
                lens_domain_new = np.reshape(dVs_new, (self.nEly_lens, self.nElx_lens))
                _ = self.dis_heat.FEM_sol_heat(lens_domain_new, self.filThr_con)
                FOM_new = self.dis_heat.compute_FOM()
                sens [i] = (FOM_new - FOM_0) / delta_dV
                print(FOM_new)
                dVs_new [i] -= delta_dV

            plot_sens_heat(self.dis_heat, sens)

        if obj == "part":

            part_domain = np.reshape(self.dVs_part, (self.nEly_part, self.nElx_part))

            sens = np.zeros(len(part_domain.flatten()))
            _ = self.dis_heat_part.FEM_sol_heat(part_domain, self.filThr_con_part)
            FOM_0 = self.dis_heat_part.compute_FOM()
            dVs_new = self.dVs_part

            for i in range(len(self.dVs_part)):
                dVs_new [i] += delta_dV
                part_domain_new = np.reshape(dVs_new, (self.nEly_part, self.nElx_part))
                _ = self.dis_heat_part.FEM_sol_heat(part_domain_new, self.filThr_con_part)
                FOM_new = self.dis_heat_part.compute_FOM()
                sens [i] = (FOM_new - FOM_0) / delta_dV
                print(FOM_new)
                dVs_new [i] -= delta_dV

            plot_sens_heat(self.dis_heat_part, sens)


        return sens

    def plot_FOM(self):
        """
        Function to plot the FOM after a given simulation.
        """
        plot_intensity(self.dis_0)

    def plot_heat(self, obj):
        """
        Function to plot the FOM after a given simulation.
        """
        if obj == "lens":
            plot_heat(self.dis_heat)
        if obj == "part":
            plot_heat(self.dis_heat_part)

    def calculate_forces(self):
        """
        Function that gives the value of the forces on the particle.
        """
        Fx = self.dis_0.Fx
        Fy = self.dis_0.Fy
        print("Fx: ", Fx)
        print("Fy: ", Fy)

    def plot_H_field(self, comp):

        plot_H_comp(self.dis_0, comp)

    
    def plot_material_interpolation(self):
        """
        Function to plot the material interpolation after a given simulation.
        """
        plot_mi(self.dis_0)

    def plot_material_interpolation_heat(self):
        """
        Function to plot the material interpolation after a given simulation.
        """
        plot_mi_heat(self.dis_heat_part)

    def plot_sensitivities(self, which, obj=None):
        """
        Function to plot the sensitivities after a given simulation.
        """
        if which == "lens":
            sens = self.sens
            plot_sens(self.dis_0, self.sens)
        if which == "part":
            sens = self.sens_part
            plot_sens_part(self.dis_0, self.sens_part)

        if which == "heat":
            if obj == "lens":
                sens = self.sens_heat
                plot_sens_heat(self.dis_heat, self.sens_heat)

            if obj == "part":
                sens = self.sens_heat_part
                plot_sens_heat(self.dis_heat_part, self.sens_heat_part)
        
        return sens

    def iteration_history(self, it_num, save=False, dir=None):

        print("----------------------------------------------")
        print("Iteration history")
        print("----------------------------------------------")

        plot_it_history(self.maxItr, self.FOM_list, self.opt.cons_1_it, self.opt.cons_1_it, it_num, save, dir)
        
