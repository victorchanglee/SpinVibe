import numpy as np

def fourth(my_system, rhsfunc, Ufunc, measure=False, timer=None, timer2=None):
     """
     fourth order Runge-Kutta formula 
     rhsfunc : R.H.S or first derivative of y. dy/dt = f(y)
     
     To Do: Addapt for new code
    
     """

   
     t_list = my_system.tlist
     if not t_list.any():
        print( '  \n Time list is empty! Exit program...')
        import sys
        sys.exit()

     dt = t_list[1]-t_list[0]  
     # main loop
     for it, t in enumerate(t_list[:-1]):
   
        Gtmp = my_system.get_my_lessG()

        k1 = rhsfunc(my_system, Ufunc(t))
        my_system.add_lessG(k1*dt/2)

        k2 = rhsfunc(my_system, Ufunc(t+dt/2))
        my_system.set_my_lessG( Gtmp+k2*dt/2 )

        k3 = rhsfunc(my_system, Ufunc(t+dt/2))
        my_system.set_my_lessG( Gtmp+k3*dt )

        k4 = rhsfunc(my_system, Ufunc(t+dt))
        my_system.set_my_lessG( Gtmp + dt/6*(k1 + 2*k2 + 2*k3 + k4) )

    
     return
