from .MMA import mmasub
import numpy as np


class optimizer:

    def __init__(self,
                 m,
                 n,
                 p,
                 xmin,
                 xmax,
                 f0, 
                 f,
                 a0,
                 a, 
                 c,
                 d,
                 maxiter,
                 move,
                 maxiniter = 0,
                 type_MMA = "MMA"
                 ):

        """Initializing optimizer.
        @ m : Number of constraints.
        @ n : Number of variables. 
        @ p : Number of objective functions to be used in minmax
        @ xmin: Lower bound on design variables
        @ xmax: Upper bound on design variables
        @ f0: FOM function.
        @ f: array with contraint functions.
        @ a0: MMA parameter.
        @ a: MMA parameter array.
        @ c: MMA parameter array. 
        @ d: MMA parameter array.
        @ maxiter: Maximum number of iterations in the optimization.
        @ move: Step size in optimization. Smaller move limits will lead to slower but steadier convergence, while larger move limits might overshoot or lead to instability.
        @ maxiniter: Maximum number of iner iterations for GCMMA. To be implemented.
        @ type_MMA: MMA or GCMMA. Last to be implemented.
        """

        self.m = m
        self.n = n
        self.p = p 
        self.xmin = xmin
        self.xmax = xmax
        self.f0 = f0 # objective function
        self.f = f # constraint function
        self.a0 =  a0
        self.a = a
        self.c = c
        self.d = d
        self.maxiter = maxiter
        self.move = move
        self.maxiniter = maxiniter
        self.type_MMA = type_MMA
        self.FOM_it = np.zeros(self.maxiter)
        self.cons_it = np.zeros(self.maxiter)
        self.lam_array = np.zeros((m, self.maxiter)) # an array with values that point at which constraints are active and which not.
        self.dVhist = np.zeros((n, maxiter))


    def optimize(self, x):

        xval = x # initial guess 
        xold1 = x
        xold2 = x 
        fval = np.zeros(self.m)[:,np.newaxis]
        dfdx = np.zeros((self.m, self.n))
        low = np.zeros(self.n)[:,np.newaxis]
        upp = np.zeros(self.n)[:,np.newaxis]


        if self.type_MMA == "MMA":
            
            for i in range(self.maxiter):

                print("----------------------------------------------")
                print("Optimization iteration: ",i)

                f0val, df0dx = self.f0(xval,i)
                self.FOM_it [i]= f0val

                for j in range(len(self.f)): 
                    fval[j], dfdx[j, :] = self.f[j](xval)
                    if j == 0:
                        self.cons_it [i]= fval [j] # saves the value of the constraint in every iteration
            
                xval_new, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(self.m,self.n,i,xval,self.xmin, self.xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,self.a0,self.a,self.c,self.d,self.move)
            
                # we update the old values of the design variables

                xold2 = xold1
                xold1 = xval
                xval = xval_new

                # we can save the lambda values, if we want to check which constraints are active or inactive.

                self.lam_array[:, i] = lam.flatten()


        return xval, self.FOM_it, self.cons_it




