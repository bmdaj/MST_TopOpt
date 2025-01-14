from .element_matrices import element_matrices
from .material_interpolation import material_interpolation_heat
import scipy
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import use_solver
import numpy as np
import time
from .filter_threshold import filter_threshold

class dis_heat:
    "Class that describes the discretized FEM model"
    def __init__(self, 
                 scaling,
                 nElx,
                 nEly):
        """
        @ scaling: scale of the physical problem; i.e. 1e-9 for nm.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ tElmIdx: Target element's index for FOM calculation.
        @ dVElmIdx: Indexes for the design variables.
        """

        self.scaling = scaling
        self.nElx = nElx
        self.nEly = nEly
        # -----------------------------------------------------------------------------------
        # INITIALIZE ELEMENT MATRICES
        # ----------------------------------------------------------------------------------- 
        LEM, MEM = element_matrices(scaling) 
        self.LEM = LEM
        self.MEM = MEM


    def index_set(self, obj):
        
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

        self.node_nrs = np.reshape(np.arange(0,self.nodesX * self.nodesY), (self.nodesY,self.nodesX)) # node numbering matrix
        self.node_nrs_flat = self.node_nrs.flatten() 

        self.elem_nrs = np.reshape(self.node_nrs[:-1,:-1], (nEY,nEX)) # element numbering matrix
        self.elem_nrs_flat = self.elem_nrs.flatten()


        self.edofMat = np.tile(self.elem_nrs_flat, (4,1)).T + np.ones((nEY*nEX,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1)) # DOF matrix: nodes per element

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

        self.nBC = np.concatenate([self.n1BC, self.n2BC, self.n3BC, self.n4BC])

        if obj == "lens":
            self.nBC_const_heat = np.concatenate([self.n4BC]) # since we want to connect the metalens design to the substrate

        if obj =="part":
            center = int(self.nodesX*self.nodesY/2)
            self.nBC_const_heat = np.array([center]) # if we want to connect the boundary to the center
            #self.nBC_const_heat = np.concatenate([self.n4BC]) # if we want to connect the boundary to the bottom

        # For the implementation of the BC into the global system matrix we need to know which nodes each boundary line has:

        self.nodes_line1 = np.tile(self.n1BC[:-1], (2,1)).T + np.ones((len(self.n1BC)-1,2))*np.tile(np.array([0, 1]), (len(self.n1BC)-1, 1))
        self.nodes_line2 = np.tile(self.n2BC[:-1], (2,1)).T + np.ones((len(self.n2BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n2BC)-1, 1))
        self.nodes_line3 = np.tile(self.n3BC[:-1], (2,1)).T + np.ones((len(self.n3BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n3BC)-1, 1))
        self.nodes_line4 = np.tile(self.n4BC[:-1], (2,1)).T + np.ones((len(self.n4BC)-1,2))*np.tile(np.array([0,1]), (len(self.n4BC)-1, 1))

        self.lines = np.arange(0, 2 * (self.nElx + self.nEly))
        self.nodes_per_line = np.concatenate([self.nodes_line1,self.nodes_line2,self.nodes_line3,self.nodes_line4]) 

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

        

    def system_RHS(self, HS_dFPST):
        """
        Sets the system's RHS.
        In this case, we count om having an incident plane wave from the RHS.
        @ HS_dFPST:  Heat source as defined by the material interpolation (design variable dependent).
        """    

        F =  np.zeros((self.nodesY*self.nodesX,1), dtype="complex128") # system RHS
        
        HS_dFPST_flat = HS_dFPST.flatten()
        
        for e, nodes in enumerate(self.edofMat):
            F[nodes.astype(int)] += 0.25*HS_dFPST_flat[e] # we have 4 nodes, so we get a fourth of the contribution

        # -----------------------------------------------------------------------------------
        # MODIFY RHS FOR DIRICHLET BOUNDARY CONDITIONS AT CONSTANT TEMPERATURE BOUNDARY
        # ----------------------------------------------------------------------------------- 

        F [self.nBC_const_heat] = 0.0 # we fix the temperature to zero at the constant temperature boundary
        I = scipy.sparse.identity(n=(len(self.node_nrs_flat)), format ="csr", dtype='complex128')
        values = np.ones_like(self.nBC_const_heat)
        N = I - scipy.sparse.csr_matrix((values,(self.nBC_const_heat.astype(int), self.nBC_const_heat.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        # apply dirichlet 0 boundary conditions with operations
        self.S = N.T @ self.S @ N + I - N 

        return F 
    
    def assemble_matrix(self, L, k):
        """
        Assembles the global system matrix.
        Since all our elements are linear rectangular elements we can use the same element matrices for all of the elements.
        @ L: Laplace matrix for each element
        @ k: wavevector of the problem (Frequency domain solver).
        """ 

        L_S = np.tile(L.flatten(),self.nElx*self.nEly) # create 1D system Laplace array
        k_S = np.repeat(k, 16)
        self.vS = k_S * L_S
        # we can take all these values and assign them to their respective nodes
        S = scipy.sparse.csr_matrix((self.vS,(self.iS.astype(int), self.jS.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype="complex128")
        # we sum all duplicates, which is equivalent of accumulating the value for each node
        S.sum_duplicates()
        S.eliminate_zeros()

        return S
    
    def solve_sparse_system(self,F):
        """
        Solves a sparse system of equations using LU factorization.
        @ S: Global System matrix.
        @ F: RHS array.
        """

        lu = sla.splu(self.S)
        T = lu.solve(F)

        return lu, T

    
    def FEM_sol_heat(self, dFP, filThr_con):
        """
        Gives the solution to the forward FEM problem; this is, the electric field solution.
        @ dFP: Design variables in the simulation domain
        @ filThr_con: Filtering and thresholding class object for the connectivity (artificial heat) problem.
        """ 
        # -----------------------------------------------------------------------------------
        # FILTERING AND THRESHOLDING ON THE MATERIAL
        # -----------------------------------------------------------------------------------
        # 

        self.filThr_con = filThr_con

        # Betas (threshold sharpness) for the conductivity and heat generation thresholding

        beta_CF = 10 # conductivity
        beta_S = 10  # heat generation

        # Threshold value for the conductivity and heat generation thresholding

        eta_CF = 0.7 # conductivity
        eta_S = 0.7  # heat generation

        self.filThr_CF =  filter_threshold(self.filThr_con.fR, self.nElx, self.nEly, eta_CF, beta_CF) 
        self.filThr_S =  filter_threshold(self.filThr_con.fR, self.nElx, self.nEly, eta_S, beta_S) 
        
        self.dFP = dFP
        self.dFPS = self.filThr_con.density_filter(np.ones((self.nEly, self.nElx)), self.filThr_con.filSca, self.dFP, np.ones((self.nEly, self.nElx)))
        self.dFPST = self.filThr_con.threshold(self.dFPS)

        # -----------------------------------------------------------------------------------
        # MATERIAL INTERPOLATION
        # ----------------------------------------------------------------------------------- 

        c_mat = 1E10 # material conductivity
        c_bck = 1E-6 # background conductivity

        S_mat = 1E20 # material heat generation
        S_bck = 0.0  # background heat generation

        self.A_C, self.dAdx_C = material_interpolation_heat(c_mat, c_bck, self.dFPST) # conductivity interpolation

        self.A_S, self.dAdx_S = material_interpolation_heat(S_mat, S_bck, self.dFPST) # heat generation interpolation
        
        # -----------------------------------------------------------------------------------
        # ASSEMBLY OF GLOBAL SYSTEM MATRIX
        # -----------------------------------------------------------------------------------

        self.S = self.assemble_matrix(self.LEM, self.A_C)

        # -----------------------------------------------------------------------------------
        # SYSTEM RHS
        # -----------------------------------------------------------------------------------

        F = self.system_RHS(self.A_S)

        # -----------------------------------------------------------------------------------
        # SOLVE SYSTEM OF EQUATIONS
        # -----------------------------------------------------------------------------------

        self.lu, self.T = self.solve_sparse_system(F)

        return self.T

    def get_lu_factorization_matrices(self):
        """
        Gives the LU factorization of a sparse matrix.
        Definitions from reference (scipy.sparse.linalg.SuperLU documentation), adjusted to case.
        """ 
        L = self.lu.L
        U = self.lu.U

        PR = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY, dtype="complex128"), (self.lu.perm_r, np.arange(self.nodesX*self.nodesY))), dtype="complex128") # Row permutation matrix
        PC = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY, dtype="complex128"), (np.arange(self.nodesX*self.nodesY), self.lu.perm_c)), dtype="complex128") # Column permutatio

        return L, U, PR, PC

    def compute_sensitivities(self, L, dAdx, T, AdjLambda):
        """
        Computes the sensitivities for all of the elements in the simulation domain.
        @ L: Laplace matrix for each element
        @ dAdx: derivative of the design variables in the simulation domain. 
        @ T: temperaure field calculated from the forward problem.
        @ AdjLambda: Vector obtained by solving S.T * AdjLambda = AdjRHS
        """ 

        sens = np.zeros(self.nEly * self.nElx)
        sens_RHS = np.zeros(self.nEly * self.nElx) # this problem also has sensitivities from the RHS, since the heat generation depends on the design!
       
        for i in range(len(self.edofMat)):

            dSdx_e = dAdx.flatten()[i] * L
            dFdx_e =  0.25*self.dAdx_S.flatten()[i]*np.ones(4) # RHS sensitivity per element

            # We also make sure that the Dirichlet (constant heat) boundary conditions are taken into account

            for j in range(len(self.edofMat[i].astype(int))):
                n = self.edofMat[i].astype(int) [j]
                if n in self.nBC_const_heat:
                    dSdx_e [j, :]  = 0
                    dSdx_e [:, j]  = 0
                    dSdx_e [j,j]  = 1
                    dFdx_e [j] = 0 

            AdjLambda_e = np.array([AdjLambda[n] for n in self.edofMat[i].astype(int)])
            T_e = np.array([T[n] for n in self.edofMat[i].astype(int)]).flatten()

            sens [i] = np.real((AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e))) [0]) # LHS sensitivities as in the EM problem 
            sens_RHS [i] = np.real((AdjLambda_e[np.newaxis] @ (-dFdx_e)) [0]) # RHS sensitivities due to the design dependent heat generation
        
        return np.reshape(sens, (self.nEly, self.nElx)), np.reshape(sens_RHS, (self.nEly, self.nElx)) # sensitivities are given separately, since they (may) rely on different filters and thresholds!


    
    def compute_FOM(self):
        """
        Computes the numerical value of the FOM.
        """ 
        
        FOM = (1/30.0)*np.log10(np.sum(self.T) / (self.nElx*self.nEly*self.scaling**2)) # normalized integral of CF summing over the nodes!
        print("FOM: ", FOM)

        return FOM



    def objective_grad(self, dVs, filThr_con):
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

        self.T = self.FEM_sol_heat(dVs, filThr_con)

        # -----------------------------------------------------------------------------------
        # COMPUTE THE FOM
        # -----------------------------------------------------------------------------------

        FOM = self.compute_FOM()

        # -----------------------------------------------------------------------------------
        #  ADJOINT OF RHS
        # -----------------------------------------------------------------------------------

        AdjRHS = np.real(np.ones_like(self.T, dtype="complex128")) / (self.nodesX*self.nodesY) 

        AdjRHS = ((1/30.0)/(np.sum(self.T) / (self.nodesX*self.nodesY)*np.log(10))) * AdjRHS # normalization as in the FOM, and derivative of log10

        # -----------------------------------------------------------------------------------
        #  SOLVE THE ADJOINT SYSTEM: S.T * AdjLambda = AdjRHS
        # -----------------------------------------------------------------------------------

        self.L, self.U, self.PR, self.PC = self.get_lu_factorization_matrices()
        AdjLambda  = self.PR.T @  sla.spsolve(self.L.T, sla.spsolve(self.U.T, self.PC.T @ (-AdjRHS)))        

        # -----------------------------------------------------------------------------------
        #  COMPUTE SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        self.sens, self.sens_RHS = self.compute_sensitivities(self.LEM, self.dAdx_C, self.T, AdjLambda)
        
        # -----------------------------------------------------------------------------------
        #  FILTER  SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        DdFSTDFS = self.filThr_con.deriv_threshold(self.dFPS)

        self.sens = self.filThr_con.density_filter(self.filThr_con.filSca, np.ones((self.nEly,self.nElx)),self.sens,DdFSTDFS)
        self.sens_RHS = self.filThr_con.density_filter(self.filThr_con.filSca, np.ones((self.nEly,self.nElx)),self.sens_RHS,DdFSTDFS)
        
        sensFOM = self.sens + self.sens_RHS # now we can finally sum both sensitivites

        # -----------------------------------------------------------------------------------
        #  FOM FOR MINIMIZATION
        # -----------------------------------------------------------------------------------

        # Plotting and printing per optimization iteration
        end = time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("Elapsed time in iteration: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")

        return FOM, sensFOM

        