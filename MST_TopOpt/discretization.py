from .element_matrices import element_matrices, boundary_element_matrix, field_gradient
from .material_interpolation import material_interpolation_sc
import scipy
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import use_solver
import numpy as np
from .plot import plot_iteration
import time
from .optomechanics import find_boundaries_projection, calc_MST, calc_F
from .adjoint import calc_AdjRHS

class dis:
    "Class that describes the discretized FEM model"
    def __init__(self, 
                 scaling,
                 nElx,
                 nEly,
                 tElmIdx,
                 dVElmIdx,
                 dVElmIdx_part,
                 dVElmIdx_part_pad):
        """
        @ scaling: scale of the physical problem; i.e. 1e-9 for nm.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ tElmIdx: Target element's index for FOM calculation.
        @ dVElmIdx: Indexes for the design variables for the metalens.
        @ dVElmIdx_part: Indexes for the design variables for the particle.
        @ dVElmIdx_part_pad: Indexes for a padded version of the design variables for the particle.
        """

        self.scaling = scaling
        self.nElx = nElx
        self.nEly = nEly
        self.tElmIdx = tElmIdx 
        self.dVElmIdx = dVElmIdx 
        self.dVElmIdx_part = dVElmIdx_part
        self.dVElmIdx_part_pad = dVElmIdx_part_pad
        
        # -----------------------------------------------------------------------------------
        # INITIALIZE ELEMENT MATRICES
        # ----------------------------------------------------------------------------------- 
        LEM, MEM = element_matrices(scaling) 
        self.LEM = LEM
        self.MEM = MEM


    def index_set(self):
        
        """
        Sets indexes for:
        a) The system matrix: self.S
        b) The boundary conditions, self.n1BC, self.n2BC, self.n3BC, self.n4BC, self.nodes_per_line
        c) The right hand side (RHS)
        d) The full node matrix (where shared nodes are treated independently) used in the sensitivity calculation
        e) The sensitivity matrix which takes into account which nodes correspond to which elements in the indexing"
        """
        
        # -----------------------------------------------------------------------------------
        # A) SET INDEXES FOR THE SYSTEM MATRIX
        # ----------------------------------------------------------------------------------- 

        nEX = self.nElx # Number of elements in X direction
        nEY = self.nEly # Number of elements in Y direction

        self.nodesX = nEX + 1 # Number of nodes in X direction
        self.nodesY = nEY + 1 # Number of nodes in Y direction

        node_nrs = np.reshape(np.arange(0,self.nodesX * self.nodesY), (self.nodesY,self.nodesX)) # node numbering matrix
        self.node_nrs = node_nrs
        self.node_nrs_flat = node_nrs.flatten() 

        self.elem_nrs = np.reshape(node_nrs[:-1,:-1], (nEY,nEX)) # element numbering matrix
        elem_nrs_flat = self.elem_nrs.flatten()

        self.edofMat = np.tile(elem_nrs_flat, (4,1)).T + np.ones((nEY*nEX,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1)) # DOF matrix: nodes per element

        # to get all the combinations of nodes in elements we can use the following two lines:

        self.iS = np.reshape(np.kron(self.edofMat,np.ones((4,1))), 16*self.nElx*self.nEly) # nodes in one direction
        self.jS = np.reshape(np.kron(self.edofMat,np.ones((1,4))), 16*self.nElx*self.nEly) # nodes in the other direction
        
        # -----------------------------------------------------------------------------------
        # B) SET INDEXES FOR THE BOUNDARY CONDITIONS
        # ----------------------------------------------------------------------------------- 

        end = self.nodesX * self.nodesY # last node number

        self.n1BC = np.arange(0,self.nodesX) # nodes top
        self.n2BC = np.arange(0,end-self.nodesX+1, self.nodesX) #left
        self.n3BC = np.arange(self.nodesX-1,end, self.nodesX) #right
        self.n4BC = np.arange(end-self.nodesX,end) #bottom

        # For the implementation of the BC into the global system matrix we need to know which nodes each boundary line has:

        nodes_line1 = np.tile(self.n1BC[:-1], (2,1)).T + np.ones((len(self.n1BC)-1,2))*np.tile(np.array([0, 1]), (len(self.n1BC)-1, 1))
        nodes_line2 = np.tile(self.n2BC[:-1], (2,1)).T + np.ones((len(self.n2BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n2BC)-1, 1))
        nodes_line3 = np.tile(self.n3BC[:-1], (2,1)).T + np.ones((len(self.n3BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n3BC)-1, 1))
        nodes_line4 = np.tile(self.n4BC[:-1], (2,1)).T + np.ones((len(self.n4BC)-1,2))*np.tile(np.array([0,1]), (len(self.n4BC)-1, 1))

        self.lines = np.arange(0, 2 * (self.nElx + self.nEly))
        self.nodes_per_line = np.concatenate([nodes_line1,nodes_line2,nodes_line3,nodes_line4])

         # to get all the combinations of nodes in lines we can use the following two lines:

        self.ibS = np.reshape(np.kron(self.nodes_per_line,np.ones((2,1))), 8*(self.nElx+self.nEly))
        self.jbS = np.reshape(np.kron(self.nodes_per_line,np.ones((1,2))), 8*(self.nElx+self.nEly))
 

        # -----------------------------------------------------------------------------------
        # C) SET INDEXES FOR THE RHS
        # ----------------------------------------------------------------------------------- 

        RHSB = self.n4BC # we select the boundary corresponding to the RHS
        self.nRHS1 = RHSB[1:]  #shared nodes
        self.nRHS2 = RHSB[:-1] #shared nodes

        self.nRHS = np.array([self.nRHS1, self.nRHS2])

        # -----------------------------------------------------------------------------------
        # D) SET INDEXES FOR THE FULL NODE MATRIX
        # ----------------------------------------------------------------------------------- 

        # to match all elements with nodes (and vice versa) we flatten the DOF matrix

        self.idxDSdx = self.edofMat.astype(int).flatten()

        # to get all the combinations of nodes in elements we can use the following two lines:

        ima0 = np.tile([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3],(self.nElx*self.nEly)) 
        jma0 = np.tile([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],(self.nElx*self.nEly))

        addTMP = np.reshape(np.tile(4*np.arange(0,self.nElx*self.nEly),16),(16, self.nElx*self.nEly)).T.flatten()

        # by adding addTMP to ima0 and jma0 we can be sure that the indexes for each node are different so we get all node combinations
        # independently. This means that if there are two elements that share a node, this will not be summed in a matrix position, but
        # taken independently.

        self.iElFull = ima0 + addTMP
        self.jElFull = jma0 + addTMP

        # -----------------------------------------------------------------------------------
        # E) SET INDEXES FOR THE SENSITIVITY MATRIX
        # ----------------------------------------------------------------------------------- 

        # now we want to index all the nodes in the elements  

        self.iElSens = np.arange(0,4*self.nElx*self.nEly)
        self.jElSens = np.reshape(np.tile(np.arange(0,self.nElx*self.nEly),4),(4, self.nElx*self.nEly)).T.flatten()

        

    def system_RHS(self,waveVector):        
        """
        Sets the system's RHS.
        In this case, we count om having an incident plane wave from the RHS.
        @ wavevector:  wavevector of the problem (Frequency domain solver).
        """        
        
        F = np.zeros((self.nodesX*self.nodesY,1), dtype=complex) # system RHS
        F[self.nRHS[0,:]] = F[self.nRHS[0,:]] +1j*waveVector # plane wave
        F[self.nRHS[1,:]] = F[self.nRHS[1,:]] +1j*waveVector # plane wave

        return self.scaling*F*1E6 # scaling for units 1E6 V/m
    
    def material_distribution(self, dVs, fR):
        """
        Sets the material in the simulation domain.
        In this case, we set the air region, the substrate and the deisgn region.
        @ dVs: Values for the design variables. Not used in free-space, but can be used for metalens systems.
        @ fR: filter radius.
        """ 

        dFP = np.zeros((self.nEly, self.nElx)) # we initialize the domain to air
        dFP[int(np.floor((self.nEly-2*fR)*9/10))+fR:-fR,fR:-fR] = 0 # the design variables for the substrate are zero in free-space
        dFP[np.ix_(self.dVElmIdx[0], self.dVElmIdx[1])] = 0 # the design variables for the metalens are zero in free-space
        return dFP

    def material_distribution_part(self, dVs, part_shape, part_size, part_center):
        """
        Adds the particle in the physical domain.
        @ part_shape: parameter that controls the shape of the particle: i.e. "circle" or "square".
        @ part_size: parameter that controls the size of the particle.
        @ part_center: parameter that controls the center of the particle.
        """

        dFP = np.zeros((self.nEly, self.nElx)) # we initialize the domain to air
        dFP_pad = np.zeros((self.nEly, self.nElx)) # we initialize the domain to air
        dVini = 1.0

        if part_shape == "square":

            dFP[part_center[1]-part_size//2:part_center[1]+part_size//2, part_center[0]-part_size//2:part_center[0]+part_size//2] = dVini
            dFP_pad[part_center[1]-part_size//2-1:part_center[1]+part_size//2+1, part_center[0]-part_size//2-1:part_center[0]+part_size//2+1] = dVini
        
        elif part_shape == "circle":

            radius = part_size // 2
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 < radius**2
            dFP[part_center[1]-radius:part_center[1]+radius, part_center[0]-radius:part_center[0]+radius][mask] = dVini

        elif part_shape == "design":

            dFP[np.ix_(self.dVElmIdx_part[0], self.dVElmIdx_part[1])] = np.reshape(dVs, (len(self.dVElmIdx_part[0]), len(self.dVElmIdx_part[1])))

        elif part_shape == "pad":

            dFP[np.ix_(self.dVElmIdx_part_pad[0], self.dVElmIdx_part_pad[1])] = np.reshape(dVs, (len(self.dVElmIdx_part_pad[0]), len(self.dVElmIdx_part_pad[1])))

        return dFP, dFP_pad
    
    def assemble_matrix(self, L, M, K, k, eps):
        """
        Assembles the global system matrix.
        Since all our elements are linear rectangular elements we can use the same element matrices for all of the elements.
        @ L: Laplace matrix for each element
        @ M: Mass matrix for each element
        @ K: Boundary matrix for each line in the boundary in air
        @ K_mat: Boundary matrix for each line in the boundary in the material
        @ k: wavevector of the problem (Frequency domain solver).
        @ eps: design variables in the simulation domain. 
        """ 

        L_S = np.tile(L.flatten(),self.nElx*self.nEly) # create 1D system Laplace array
        M_S = np.tile(M.flatten(),self.nElx*self.nEly) # create 1D system Mass array
        eps_S = np.repeat(eps,16).flatten() # create 1D system design variable array    
        self.vS = L_S  - (k**2)* M_S * eps_S # values to cast to the system matrix

        # we can take all these values and assign them to their respective nodes
        S = scipy.sparse.csr_matrix((self.vS,(self.iS.astype(int), self.jS.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        # we sum all duplicates, which is equivalent of accumulating the value for each node
        S.sum_duplicates()

        # we follow a similar process for the boundaries; however, now we go through the nodes on each line in the boundaries
        self.bS = np.tile(K.flatten(),(2*self.nElx+2*self.nEly))

        K = scipy.sparse.csr_matrix((self.bS,(self.ibS.astype(int), self.jbS.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        K.sum_duplicates()

        # we sum the contribution from the boundary to global the system matrix
        S = S + K 

        return S
    
    def solve_sparse_system(self,F):
        """
        Solves a sparse system of equations using LU factorization.
        @ S: Global System matrix
        @ F: RHS array 
        """ 
        lu = sla.splu(self.S)
        Ez = lu.solve(F)
        return lu, Ez

    
    def FEM_sol(self, dVs, dVs_part, phy, filThr):
        """
        Gives the solution to the forward FEM problem; this is, the electric field solution.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        """ 
        # -----------------------------------------------------------------------------------
        # FILTERING AND THRESHOLDING ON THE MATERIAL
        # ----------------------------------------------------------------------------------- 

        # For the metalens

        self.dFP = self.material_distribution(dVs, filThr.fR)
        self.dFPS = filThr.density_filter(np.ones((self.nEly, self.nElx)), filThr.filSca, self.dFP, np.ones((self.nEly, self.nElx)))
        self.dFPST = filThr.threshold(self.dFPS)

        # For the particle:

        self.dFP_part, self.dFP_part_pad = self.material_distribution_part(dVs_part, phy.part_shape, phy.part_size, phy.part_center)
        self.dFPS_part = filThr.density_filter(np.ones((self.nEly, self.nElx)), filThr.filSca, self.dFP_part, np.ones((self.nEly, self.nElx)))
        self.dFPST_part = filThr.threshold(self.dFPS_part)

        # -----------------------------------------------------------------------------------
        # MATERIAL INTERPOLATION
        # ----------------------------------------------------------------------------------- 

        self.dFPST_total = self.dFPST + self.dFPST_part # we add the contribution of the metalens and the particle

        self.A, self.dAdx = material_interpolation_sc(phy.eps, self.dFPST_total, phy.alpha)

        # -----------------------------------------------------------------------------------
        # FIND THE NODES (AND NORMALS) IN THE BOUNDARY OF THE PARTICLE
        # ----------------------------------------------------------------------------------- 

        dVs_part_pad = np.ones(len(self.dVElmIdx_part_pad[0])*len(self.dVElmIdx_part_pad[1]))

        # Now we create the padded material distribution for the particle.

        boundary, _ = self.material_distribution_part(np.ones_like(dVs_part_pad), "pad", phy.part_size, phy.part_center)

        self.b_idx_x, self.b_idx_y, self.b_n_x, self.b_n_y, self.db_n_x, self.db_n_y = find_boundaries_projection(self, boundary, self.edofMat, self.nElx, self.nEly)
        
        # Calculate the node numbers at the boundary
        
        self.nodes_b_x = self.node_nrs [self.b_idx_x[0],self.b_idx_x[1]]
        self.nodes_b_y = self.node_nrs [self.b_idx_y[0],self.b_idx_y[1]]
    

        # -----------------------------------------------------------------------------------
        # SYSTEM RHS
        # -----------------------------------------------------------------------------------

        F = self.system_RHS(phy.k)

        # -----------------------------------------------------------------------------------
        # ASSEMBLY OF GLOBAL SYSTEM MATRIX
        # -----------------------------------------------------------------------------------

        K_s = boundary_element_matrix(self.scaling, phy.k, 1.0) #air boundary element

        self.S = self.assemble_matrix(self.LEM, self.MEM, K_s, phy.k, self.A)

        # -----------------------------------------------------------------------------------
        # SOLVE SYSTEM OF EQUATIONS
        # -----------------------------------------------------------------------------------

        self.lu, self.Ez = self.solve_sparse_system(F)

        # -----------------------------------------------------------------------------------
        # CALCULATE THE REST OF THE FIELD COMPONENTS
        # -----------------------------------------------------------------------------------

        Ez = np.reshape(self.Ez, (self.nodesY, self.nodesX))

        # Because of TE polarization: 

        self.Ex = np.zeros_like(Ez) 
        self.Ey = np.zeros_like(Ez) 
        self.Hz = np.zeros_like(Ez) 

        # Calculate the rest of the components from the gradient:

        self.dEzdx, self.dEzdy  = field_gradient(self.edofMat, self.Ez.flatten(), self.scaling)

        # From Maxwell's equations:

        self.omega = phy.omega

        self.Hx = 1j * self.dEzdy / (phy.omega*phy.mu_0)
        self.Hy = -1j * self.dEzdx / (phy.omega*phy.mu_0)

        self.Hx = np.reshape(self.Hx, (self.nodesY, self.nodesX))
        self.Hy = np.reshape(self.Hy, (self.nodesY, self.nodesX))

        # -----------------------------------------------------------------------------------
        # CALCULATE THE FORCE VIA THE MAXWELL STRESS TENSOR (MST)
        # -----------------------------------------------------------------------------------

        self.MST, self.MST_p_x, self.MST_p_y = calc_MST(self.Ex.flatten(), self.Ey.flatten(), self.Ez.flatten(), self.Hx.flatten(), self.Hy.flatten(), self.Hz.flatten())

        # Calculate the force components:

        self.Fx = calc_F(self.MST_p_x, self.nodes_b_x, self.nodes_b_y,  self.b_n_x, self.b_n_y, self.scaling)
        self.Fy = calc_F(self.MST_p_y, self.nodes_b_x, self.nodes_b_y, self.b_n_x, self.b_n_y, self.scaling)

        self.compute_FOM()
        
        return self.Ez

    def get_lu_factorization_matrices(self):
        """
        Gives the LU factorization of a sparse matrix.
        Definitions from reference (scipy.sparse.linalg.SuperLU documentation), adjusted to case.
        """ 
        L = self.lu.L
        U = self.lu.U
        PR = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY), (self.lu.perm_r, np.arange(self.nodesX*self.nodesY))), dtype=complex) # Row permutation matrix
        PC = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY), (np.arange(self.nodesX*self.nodesY), self.lu.perm_c)), dtype=complex)# Column permutation matrix

        return L, U, PR, PC

    def compute_sensitivities(self, M, k, dAdx, Ez, AdjLambda):
        """
        Computes the sensitivities for all of the elements in the simulation domain.
        @ M: Mass matrix for each element
        @ k: wavevector of the problem (Frequency domain solver).
        @ dAdx: derivative of the design variables in the simulation domain. 
        @ Ez: electric field calculated from the forward problem.
        @ AdjLambda: Vector obtained by solving S.T * AdjLambda = AdjRHS
        """ 

        sens = np.zeros(self.nEly * self.nElx)
        sens_part = np.zeros(self.nEly * self.nElx)

        for i in range(len(self.edofMat)):
            
            dSdx_e = - (k**2)* M * dAdx.flatten()[i] # derivative of the system matrix

            AdjLambda_e = np.array([AdjLambda[n] for n in self.edofMat[i].astype(int)]) #adjoint per element
            Ez_e = np.array([Ez[n] for n in self.edofMat[i].astype(int)]).flatten() #field solution per element

            sens [i] = 2*np.real((AdjLambda_e[np.newaxis] @ dSdx_e @ Ez_e) [0]) # sensitivity for metalens
            sens_part [i] = 2*np.real((AdjLambda_e[np.newaxis] @ dSdx_e @ Ez_e) [0]) # sensitivity for particle 

            # Note: Sensitivity for metalens and particle is the same, but that might change if the we consider the particle boundaries
            # as the bounding box for the force calculation, now we are just considering a surrounding box with the size of the 
            # design domain.

        return np.reshape(sens, (self.nEly, self.nElx)), np.reshape(sens_part, (self.nEly, self.nElx))

    
    def compute_FOM(self):
        """
        Computes the numerical value of the FOM.
        """ 
        self.FOM = (self.Fy)/2.903E-6 # normalized by the initial value in the optimization

        return self.FOM

    def objective_grad(self, dVs, dVs_part, phy, filThr):
        """
        Evaluates the FOM via the forward FEM problem and calculates the design sensitivities.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        """ 
        
        start = time.time() # Measure the time to compute elapsed time when finished
        
        # -----------------------------------------------------------------------------------
        # SOLVE FORWARD PROBLEM
        # -----------------------------------------------------------------------------------

        self.Ez = self.FEM_sol(dVs, dVs_part, phy, filThr) 
        print('FOM: ', self.FOM)

        # -----------------------------------------------------------------------------------
        # COMPUTE THE FOM
        # -----------------------------------------------------------------------------------

        FOM = self.compute_FOM()
        
        # -----------------------------------------------------------------------------------
        #  ADJOINT OF RHS
        # -----------------------------------------------------------------------------------

        AdjRHS = calc_AdjRHS(self, phy)

        # ----------------------------------------------------------------------------------
        #  SOLVE THE ADJOINT SYSTEM: S.T * AdjLambda = AdjRHS
        # -----------------------------------------------------------------------------------

        L, U, PR, PC = self.get_lu_factorization_matrices()
        AdjLambda  = PR.T @  scipy.sparse.linalg.spsolve(L.T, scipy.sparse.linalg.spsolve(U.T, PC.T @ (-0.5*AdjRHS)))
        
        # -----------------------------------------------------------------------------------
        #  COMPUTE SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        self.sens, self.sens_part = self.compute_sensitivities(self.MEM, phy.k, self.dAdx, self.Ez, AdjLambda)

        # -----------------------------------------------------------------------------------
        #  FILTER AND THRESHOLD SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        # We need to filter the sensitivities following the chain rule

        DdFSTDFS = filThr.deriv_threshold(self.dFPS)
        sensFOM = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx)),self.sens,DdFSTDFS)

        # We need to threshold the sensitivities following the chain rule

        DdFSTDFS_part = filThr.deriv_threshold(self.dFPS_part)
        sensFOM_part = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx)),self.sens_part,DdFSTDFS_part)

        # -----------------------------------------------------------------------------------
        #  SENSITIVITIES FOR DESIGN REGION
        # -----------------------------------------------------------------------------------

        sensFOM = sensFOM[np.ix_(self.dVElmIdx[0], self.dVElmIdx[1])]

        sensFOM_part = sensFOM_part[np.ix_(self.dVElmIdx_part[0], self.dVElmIdx_part[1])]

        # -----------------------------------------------------------------------------------
        #  FOM FOR MINIMIZATION
        # -----------------------------------------------------------------------------------

        FOM = - FOM
        sensFOM = - sensFOM
        sensFOM_part = - sensFOM_part

        # Plotting and printing per optimization iteration
        end = time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("Elapsed time in iteration: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
        plot_iteration(self)
        
        return FOM, sensFOM/2.903E-6, sensFOM_part/2.903E-6 # normalized by the initial value in the optimization
        