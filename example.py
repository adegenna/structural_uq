import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint , solve_ivp


def rhs_vdp( t , u , eps=0 ):

    # u = [ u_1 , v_1 , ... , u_N , v_N ]

    udot = np.zeros( 2 )

    udot[0] = u[1]
    udot[1] = -u[0] + eps * ( 1 - u[0]**2 ) * u[1]

    return udot


def rhs_vdp_pexpansion_O0( t , u ):

    # u = [ u_1 , v_1 , ... , u_N , v_N ]

    udot = np.zeros( 2 )

    udot[0] = u[1]
    udot[1] = -u[0]

    return udot


def rhs_vdp_pexpansion_O1( t , u ):

    # u = [ u_1 , v_1 , ... , u_N , v_N ]

    udot = np.zeros( 4 )

    udot[0] = u[1]
    udot[1] = -u[0]

    udot[2] = u[3]
    udot[3] = -u[2] + ( 1 - u[0]**2 ) * u[1]

    return udot


def rhs_vdp_pexpansion_O2( t , u ):

    # u = [ u_1 , v_1 , ... , u_N , v_N ]

    udot = np.zeros( 6 )

    udot[0] = u[1]
    udot[1] = -u[0]

    udot[2] = u[3]
    udot[3] = -u[2] + ( 1 - u[0]**2 ) * u[1]

    udot[4] = u[5]
    udot[5] = -u[4] + ( 1 - u[0]**2 ) * u[3] - 2*u[0]*u[2]*u[1]

    return udot


def main():

    t = np.linspace( 0 , 0.25 , 100 )

    nx = 20
    nsamps = nx**2
    #uv0_O0 = np.random.uniform( [-3,-3] , [3,3] , [ nsamps , 2 ] )
    uu,vv = np.meshgrid( np.linspace(-2,2,nx) , np.linspace(-2,2,nx) )
    uv0_O0 = np.vstack( [ uu.ravel() , vv.ravel() ] ).T
    uv0_O1 = np.zeros( [ nsamps , 4 ] )
    uv0_O2 = np.zeros( [ nsamps , 6 ] )
    uv0_O1[:,0:2] = uv0_O0.copy()
    uv0_O2[:,0:2] = uv0_O0.copy()

    eps = 0.5

    E0 = np.zeros( nsamps )
    E1 = np.zeros( nsamps )
    E2 = np.zeros( nsamps )

    for i in range( nsamps ):

        uv_O0 = solve_ivp( rhs_vdp_pexpansion_O0 , [0,1] , uv0_O0[i] , method="RK45" , dense_output=True).sol(t).T
        uv_O1 = solve_ivp( rhs_vdp_pexpansion_O1 , [0,1] , uv0_O1[i] , method="RK45" , dense_output=True).sol(t).T
        uv_O2 = solve_ivp( rhs_vdp_pexpansion_O2 , [0,1] , uv0_O2[i] , method="RK45" , dense_output=True).sol(t).T
        uv    = solve_ivp( lambda x,y : rhs_vdp(x,y,eps=eps) , [0,1] , uv0_O0[i] , method="RK45" , dense_output=True).sol(t).T
        
        app_0 = uv_O0[:,0:2]
        app_1 = uv_O1[:,0:2] + eps * uv_O1[:,2:4]
        app_2 = uv_O2[:,0:2] + eps * uv_O2[:,2:4] + eps**2 * uv_O2[:,4:6]

        E0[i] = np.linalg.norm( uv - app_0 )**2
        E1[i] = np.linalg.norm( uv - app_1 )**2
        E2[i] = np.linalg.norm( uv - app_2 )**2

        plt.figure(1)
        plt.plot( uv[:,0] , uv[:,1] , 'k' )
        plt.plot( app_0[:,0] , app_0[:,1] , 'b--' )
        #plt.plot( app_1[:,0] , app_1[:,1] , 'g--' )
        plt.plot( app_2[:,0] , app_2[:,1] , 'r--' )


    plt.figure()
    plt.hist( E0 , bins=32 , color='b' , alpha=0.5 )
    plt.hist( E1 , bins=32 , color='g' , alpha=0.5 )
    plt.hist( E2 , bins=32 , color='r' , alpha=0.5 )

    plt.show()


if __name__ == '__main__':

    main()

