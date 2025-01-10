from MMA import mmasub, asymp, gcmmasub, concheck, raaupdate, kktcheck
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
        @ x : Input variables (guess).
        m
        """

        self.m = m
        self.n = n
        self.p = p # number of objective functions to be used in minmax
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
        self.cons_1_it = np.zeros(self.maxiter)
        self.cons_2_it = np.zeros(self.maxiter)
        self.cons_3_it = np.zeros(self.maxiter)
        self.lam_array = np.zeros((m, self.maxiter))
        self.dVhist = np.zeros((n, maxiter))


    def optimize(self, x):

        xval = x # initial guess 
        xold1 = x
        xold2 = x 
        fval = np.zeros(self.m)[:,np.newaxis]
        fvalnew = np.zeros(self.m)[:,np.newaxis]
        dfdx = np.zeros((self.m, self.n))
        dfdxnew = np.zeros((self.m, self.n))
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
                        self.cons_1_it [i]= fval [j]
                    if j == 1: 
                        self.cons_2_it [i]= fval [j]
                    if j == 2: 
                        self.cons_3_it [outit]= fval [j]

            
                xval_new, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(self.m,self.n,i,xval,self.xmin, self.xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,self.a0,self.a,self.c,self.d,self.move)
            
                xold2 = xold1
                xold1 = xval
                xval = xval_new

                self.lam_array[:, i] = lam.flatten()


        return xval, self.FOM_it, self.cons_1_it, self.cons_2_it, self.cons_3_it




