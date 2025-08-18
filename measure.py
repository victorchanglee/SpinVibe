import numpy as np
from scipy.optimize import curve_fit
from mpi4py import MPI

class measure:
    def __init__(self,rho_t,S_operator,Mpol,tlist):
        self.rho_t = rho_t
        self.Mpol = Mpol

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
        
        # Calculate rotation matrix if Mpol is provided
        if self.Mpol is not None:
            R = self.rotation_matrix_z_to_direction(self.Mpol)
            
            # Transform spin operators according to rotation
            # New operators: S'_i = R_ij * S_j
            Sx_rot = R[0,0]*Sx + R[0,1]*Sy + R[0,2]*Sz
            Sy_rot = R[1,0]*Sx + R[1,1]*Sy + R[1,2]*Sz  
            Sz_rot = R[2,0]*Sx + R[2,1]*Sy + R[2,2]*Sz
        else:
            # Use original operators
            Sx_rot = Sx
            Sy_rot = Sy
            Sz_rot = Sz
        
        for t in range(self.tsteps):
            self.Mvec[0,t] = np.trace(np.dot(Sx_rot, rho_t[:,:,t]))
            self.Mvec[1,t] = np.trace(np.dot(Sy_rot, rho_t[:,:,t]))
            self.Mvec[2,t] = np.trace(np.dot(Sz_rot, rho_t[:,:,t]))
        
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

        target_value = 0.0001  
        tolerance = 1E-6
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
    
    def rotation_matrix_z_to_direction(self,target_direction):
        """
        Alternative: Build rotation using spherical coordinates.
        Rotates Z-axis to target_direction using two sequential rotations.
        """
        # Normalize target direction  
        target = np.array(target_direction, dtype=float)
        target = target / np.linalg.norm(target)
        
        x, y, z = target
        
        # Calculate spherical angles
        r = np.sqrt(x**2 + y**2 + z**2)  # Should be 1 after normalization
        theta = np.arccos(z / r)  # Polar angle from Z-axis
        phi = np.arctan2(y, x)    # Azimuthal angle in XY plane
        
        # First rotation: around Y-axis by theta
        Ry = np.array([[np.cos(theta),  0, np.sin(theta)],
                    [0,              1, 0           ],
                    [-np.sin(theta), 0, np.cos(theta)]])
        
        # Second rotation: around Z-axis by phi  
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi),  np.cos(phi), 0],
                    [0,            0,           1]])
        
        # Combined rotation
        R = Rz @ Ry
        
        return R