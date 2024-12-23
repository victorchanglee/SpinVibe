import numpy as np

def td_spin(my_system, Uext=None, timers=None, computeU=False):

   """
   To Do: Adapt to new code

   h_k      : mf_energy = self.mfe
   dh_k     : correction of quasi-particle energy
   U_k      : external field, amplitude at t only

   rhs = h_k + dh_k + U_k  
    
   """

   timers[0].start()
   cohsex_ham = np.array(my_system.ham)
   timers[0].end()

   # interband coupling
   if Uext.any:
      cohsex_ham += - np.einsum('a,kija->kij', Uext, my_system.get_my_dipole(inter=False))

   lessG  = my_system.get_my_lessG()
   lessG0 = my_system.get_my_lessG(equilibrium=True)

   timers[1].start()
   rhs = -1j * ( np.einsum('ijk,ikl->ijl', cohsex_ham, lessG) - np.einsum('ijk,ikl->ijl', lessG, cohsex_ham)\
                 + np.einsum('jk,ijk->ijk', my_system.Gamma, lessG - lessG0) )
   timers[1].end()

   #timers[2].start()
   ## intraband coupling
   if Uext.any:
#      rhs = rhs -1j * (-1j* Uext * my_system.get_dipole_coupling())

      rhs = rhs -1j * (-1j *np.einsum('a,kija->kij', Uext, my_system.get_dipole_coupling()))

##   #rhs = rhs -1j * (-1j* Uext * my_system.get_dipole_coupling_step())
   #timers[2].end()

   if computeU:
      return rhs, cohsex_ham
   else:
      return rhs
