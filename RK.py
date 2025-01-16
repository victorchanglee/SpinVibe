import numpy as np
from scipy.integrate import solve_ivp

class RK:
     def __init__(self,rho,R1,R2,R4,tf,tlist):
          
          self.rho = rho
          self.R1 = R1
          self.R2 = R2
          self.R4 = R4
          self.R = self.R1+self.R2+self.R4
          self.hdim = self.rho.shape[0]
          self.tf = tf
          self.trange= [0,tf]
          self.tlist = tlist
          self.tsteps = len(self.tlist)

          self.drho_dt = np.zeros([self.hdim,self.hdim,len(self.tlist)],dtype=np.complex128)

          self.drho_dt = self.runge_kutta()

          return

     def rhs(self, t, rho_vec, R, hdim):

          rho = rho_vec.reshape((hdim, hdim))
          rhs = np.zeros_like(rho, dtype=np.complex128)

          rhs = np.zeros([hdim, hdim], dtype=np.complex128)
          rhs = np.einsum('abcd,cd -> ab',R, rho)
          
          return rhs.flatten()
     
     def runge_kutta(self):

          rho_vec = self.rho.flatten()

          tmp = solve_ivp(self.rhs,
                          self.trange,
                          rho_vec,
                          t_eval=self.tlist,
                          args=(self.R,self.hdim),
                          method='RK45')

          
          drho_dt = tmp.y.reshape(self.hdim, self.hdim,self.tsteps)

          return drho_dt