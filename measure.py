import numpy as np


class measure:
    def __init__(self,drho_dt,S_operator,tlist):
        self.drho_dt = drho_dt

        self.S_operator = S_operator
        self.Sx = self.S_operator[:,:,0]
        self.Sy = self.S_operator[:,:,1]
        self.Sz = self.S_operator[:,:,2]

        self.tlist = tlist
        self.tsteps = len(self.tlist)

        self.Mvec = np.zeros([3,self.tsteps],dtype=np.complex128)

        return
    
    def magnetization(self):

        for t in range(self.tsteps):
            self.Mvec[0,t] = np.trace(np.dot(self.Sx,self.drho_dt[:,:,t]))
            self.Mvec[1,t] = np.trace(np.dot(self.Sy,self.drho_dt[:,:,t]))
            self.Mvec[2,t] = np.trace(np.dot(self.Sz,self.drho_dt[:,:,t]))
            

        return
