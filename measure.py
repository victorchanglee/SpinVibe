import numpy as np
from scipy.optimize import curve_fit
from mpi4py import MPI

class measure:
    def __init__(self,rho_t,S_operator,tlist,pol,init_type):
        self.rho_t = rho_t
        self.pol = pol
        self.S_operator = S_operator
        self.init_type = init_type
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
        """
        Calculate magnetization with rotated coordinate system.
        
        Parameters:
        Mpol : array-like, optional
            New direction for Z-axis [x, y, z]. If None, uses original [0,0,1].
            Will be automatically normalized.
        """
        Sx = np.real(self.Sx)
        Sy = np.real(self.Sy)
        Sz = np.real(self.Sz)
        
        rho_t = np.real(self.rho_t)
        
        for t in range(self.tsteps):
            self.Mvec[0,t] = np.real(np.trace(self.Sx @ rho_t[:,:,t]))
            self.Mvec[1,t] = np.real(np.trace(self.Sy @ rho_t[:,:,t]))
            self.Mvec[2,t] = np.real(np.trace(self.Sz @ rho_t[:,:,t]))

        return
    

    
    def calc_t1(self):
        """
        Compute the T1 time from the Mz(t) signal using exponential fitting.
        Returns T1 and its uncertainty.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

            # polarization axis
        n = np.array(self.pol if self.pol is not None else [0., 0., 1.], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("self.Mpol must not be the zero vector")
        n /= n_norm

        
        # Extract Mz(t) data and time points


        if self.init_type == 'polarized':
            if rank == 0:
                print("Measuring T1 along the polarization axis")
            self.Mz = n @ self.Mvec
        else:
            self.Mz = np.linalg.norm(self.Mvec, axis=0)
        
        t_data = self.tlist  # Time points (ensure this is defined in your class)

        target_value = 0.0001  
        tolerance = 1e-6  

        # Allow both positive and negative
        Mz_eq = min(self.Mz, key=lambda x: abs(x - target_value))

        for i in range(1, len(self.Mz)):
            if abs(self.Mz[i] - self.Mz[i - 1]) < tolerance:
                Mz_eq = self.Mz[i]
                break

        if rank == 0:
            print("Equilibrium M_parallel:", Mz_eq)

        # Define the T1 model function
        def Mz_model(t, Mz_initial, Mz_eq, T1):
            return (Mz_initial - Mz_eq) * np.exp(-t / T1) + Mz_eq
        
        # Initial guesses for parameters [Mz_initial, Mz_eq, T1]
        p0 = [self.Mz[0], Mz_eq, 1]  

        # Define bounds (must have length 3)
        lower = [-np.inf, -np.inf, 0.0]   # T1 must be positive
        upper = [ np.inf,  np.inf, np.inf]
        
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