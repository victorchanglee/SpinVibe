import numpy as np
from scipy.optimize import curve_fit
from mpi4py import MPI

class measure:
    def __init__(self,rho_t,S_operator,tlist):
        self.rho_t = rho_t

        self.S_operator = S_operator
        self.Sx = self.S_operator[:,:,0]
        self.Sy = self.S_operator[:,:,1]
        self.Sz = self.S_operator[:,:,2]

        self.tlist = tlist
        self.tsteps = len(self.tlist)

        self.Mvec = np.zeros([3,self.tsteps],dtype=np.float64)

        self.magnetization()

        self.T1, self.T1_err = self.calc_t1()

        return
    
    def magnetization(self):
        Sx = np.real(self.Sx)
        Sy = np.real(self.Sy)
        Sz = np.real(self.Sz)

        rho_t = np.real(self.rho_t)
        for t in range(self.tsteps):
            self.Mvec[0,t] = np.trace(np.dot(Sx,rho_t[:,:,t]))
            self.Mvec[1,t] = np.trace(np.dot(Sy,rho_t[:,:,t]))
            self.Mvec[2,t] = np.trace(np.dot(Sz,rho_t[:,:,t]))
            

        return
    

    
    def calc_t1(self):
        """
        Compute the T1 time from the Mz(t) signal using exponential fitting.
        Returns T1 and its uncertainty.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Extract Mz(t) data and time points
        self.Mz = np.real(self.Mvec[2, :])  # Assuming Mvec[2,:] is Mz(t)
        t_data = self.tlist  # Time points (ensure this is defined in your class)

        target_value = 0.001  
        tolerance = 1E-5
        lower = [-np.inf, -np.inf,    0]
        upper = [ np.inf,  np.inf, np.inf]

        positive_Mz = [x for x in self.Mz if x > 0]

        Mz_eq = min(positive_Mz, key=lambda x: abs(x - target_value))

        for i in range(1, len(self.Mz)):
            if abs(self.Mz[i] - self.Mz[i - 1]) < tolerance:
                Mz_eq = self.Mz[i]
                break
      
        if rank == 0:
            print("Equilibrium Mz:",Mz_eq)
        

        # Define the T1 model function
        def Mz_model(t, Mz_initial, Mz_eq, T1):
            return (Mz_initial - Mz_eq) * np.exp(-t / T1) + Mz_eq
        
        # Initial guesses for parameters [Mz_initial, Mz_eq, T1]
        p0 = [self.Mz[0], Mz_eq, 0.5]  
        
        # Perform the fit
        try:
            params, covariance = curve_fit(Mz_model, t_data, self.Mz, p0=p0,bounds=(lower, upper))
        except Exception as e:
            print(f"Fit failed: {e}")
            return None, None
        
        # Extract parameters and errors
        Mz_initial_fit, Mz_eq_fit, T1_fit = params
        errors = np.sqrt(np.diag(covariance))
        T1_err = errors[2]  # Uncertainty in T1
        

        
        return T1_fit, T1_err
    
