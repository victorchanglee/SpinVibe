import numpy as np
from scipy.integrate import solve_ivp

def RK(rho0, R, dt, tlist):
    nsteps = len(tlist)
    hdim = len(rho0)
    rho = np.zeros((nsteps,hdim), dtype=np.complex128)
    rho[0] = rho0.copy()  # Force rho0 to be 1D
    
    for i in range(nsteps - 1):
        # Ensure R is 2D and rho[i] is 1D
        k1 = R @ rho[i]
        k2 = R @ (rho[i] + 0.5 * dt * k1)
        k3 = R @ (rho[i] + 0.5 * dt * k2)
        k4 = R @ (rho[i] + dt * k3)
        
        rho[i+1] = rho[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return rho