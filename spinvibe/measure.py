import numpy as np
from scipy.optimize import curve_fit

# Try to import MPI, fall back to serial mode if not available
try:
    from mpi4py import MPI
    mpi_on = True
except ImportError:
    mpi_on = False

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
        self.dt = tlist[1]-tlist[0]
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
        Compute T1 using logarithmic transformation for linear fitting.
        Cuts data when Mz drops below 1e-2.
        """
        if mpi_on:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        else:
            rank = 0

        # Get Mz data
        n = np.array(self.pol if self.pol is not None else [0., 0., 1.], dtype=float)
        n /= np.linalg.norm(n)
        
        if self.init_type == 'polarized':
            self.Mz = n @ self.Mvec
        else:
            self.Mz = np.linalg.norm(self.Mvec, axis=0)
        
        t_data = self.tlist
        
        # Find where Mz drops below 10%
        cutoff_mask = self.Mz > 0.1
        cutoff_idx = np.sum(cutoff_mask)
        
        if cutoff_idx < 3:
            if rank == 0:
                print(f"Warning: Only {cutoff_idx} points above 1e-2. Using minimum of 3 points.")
            cutoff_idx = min(3, len(self.Mz))
        
        # Cut the data
        t_fit = t_data[:cutoff_idx]
        Mz_fit = self.Mz[:cutoff_idx]
        
        if rank == 0:
            print(f"Using first {cutoff_idx} of {len(self.Mz)} points (Mz > 1e-2)")
            print(f"Time range: {t_fit[0]:.4e} to {t_fit[-1]:.4e}")
            print(f"Mz range: {Mz_fit[0]:.4f} to {Mz_fit[-1]:.4f}")
        
        Mz_eq = 0.0
        
        # Calculate delta from equilibrium
        delta_Mz_fit = Mz_fit - Mz_eq
        
        # Transform to linear form: ln|Mz(t) - Mz_eq| = ln|Mz0 - Mz_eq| - t/T1
        log_delta_Mz = np.log(np.abs(delta_Mz_fit))
        
        # Linear fit: y = a + b*t where b = -1/T1
        coeffs = np.polyfit(t_fit, log_delta_Mz, 1)
        slope, intercept = coeffs
        
        T1_fit = -1.0 / slope
        
        # Estimate uncertainty from residuals
        y_pred = slope * t_fit + intercept
        residuals = log_delta_Mz - y_pred
        residual_std = np.std(residuals)
        
        # Propagate uncertainty to T1
        T1_err = T1_fit * residual_std / np.sqrt(len(t_fit))
        
        # Calculate R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_delta_Mz - np.mean(log_delta_Mz)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if rank == 0:
            print(f"R² = {r_squared:.4f}")
        
        return T1_fit, T1_err