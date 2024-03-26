# Modified advect function to emit from a point source only at the first time step


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import numpy as np

def initialize(timesteps, cr):
    ''' initialize the physical system, horizontal grid size, etc
    '''
    # below are the parameters that can be varied
    u = 20.
    domain_length = 500 
    effective_points = 500
    dx = domain_length/effective_points # 500 total points/every 500 points = 1 = dx

    # cr = 0.4 
    dt = cr * dx/ u
    Numpoints = effective_points + 1
    shift = 5 # this is x star
    c_0 = 1
    epsilon = 0.0001

# create the concentration matrix and initialize it
    cmatrix = np.zeros((timesteps+1, Numpoints+4))
    cmatrix[0, shift]= c_0 / u 

# set the boundary points
    cmatrix = boundary_conditions(cmatrix, 0, Numpoints)

    return dx, u, dt, Numpoints, epsilon, cmatrix

def boundary_conditions(cmatrix, time, Numpoints):
    '''Set boundary conditions (double thick so it work for Bott Scheme as well as central and upstream
    '''
    cmatrix[time, 0] = cmatrix[time, Numpoints-1]
    cmatrix[time, 1] = cmatrix[time, Numpoints]
    cmatrix[time, Numpoints+2] = cmatrix[time, 3]
    cmatrix[time, Numpoints+3] = cmatrix[time, 4]

    return cmatrix

def gettable(order=4):

    '''read in the corresponding coefficient table for the calculation of coefficients for advection3
    '''

# create a matrix to store the table to be read in
    temp = np.zeros(5)
    ltable = np.zeros((order + 1, 5))

    fname = f'/Users/nduboc/repos/numeric_2024/numlabs/lab10/Tables/l{4}_table.txt'
    fp = open(fname, 'r')
    for i in range(order+1):
        line = fp.readline()
        temp = line.split()
        ltable[i, :]= temp

    fp.close()
    return ltable


def step_advect(timesteps, ltable, cmatrix, order, Numpoints, u, dt, dx, epsilon):
    '''Step algorithm for Bott Scheme'''

# create a matrix to store the current coefficients a(j, k)
    amatrix = np.zeros((order+1, Numpoints))

    for timecount in range(0,timesteps):
        for base in range(0,5): # 5 is for the number of coefficients in a single time step for a given order polynomial 
            amatrix[0:order+1, 0:Numpoints] += np.dot(
                ltable[0:order+2, base:base+1],
                cmatrix[timecount:timecount+1, 0+base:Numpoints+base])

# calculate I of c at j+1/2 , as well as I at j
# as these values will be needed to calculate i at j+1/2 , as
# well as i at j

# calculate I of c at j+1/2(Iplus),
# and at j(Iatj)
        Iplus = np.zeros(Numpoints)
        Iatj = np.zeros(Numpoints)

        tempvalue= 1 - 2*u*dt/dx
        for k in range(order+1):
            Iplus += amatrix[k] * (1- (tempvalue**(k+1)))/(k+1)/(2**(k+1))
            Iatj += amatrix[k] * ((-1)**k+1)/(k+1)/(2**(k+1))
        Iplus[Iplus < 0] = 0
        Iatj = np.maximum(Iatj, Iplus + epsilon)

# finally, calculate the current concentration
        cmatrix[timecount+1, 3:Numpoints+2] = (cmatrix[timecount, 3:Numpoints+2] *(1 - Iplus[1:Numpoints]/ Iatj[1:Numpoints]) 
        +cmatrix[timecount, 2:Numpoints+1]*Iplus[0:Numpoints-1]/ Iatj[0:Numpoints-1])

# set the boundary condition at the first point
        cmatrix[timecount+1, 2]= cmatrix[timecount+1, Numpoints+1]
# set the other boundary points
        cmatrix = boundary_conditions(cmatrix, timecount+1, Numpoints)

    return cmatrix

def make_graph(cmatrix, timesteps, Numpoints, dt, cr, order):
    """Create graphs of the model results using matplotlib.
    """
    ampF = np.max(cmatrix[0,:])
    ampE = np.max(cmatrix[-1,:])
    error = abs(ampE-ampF)
    # Create a figure with size 15, 5
    fig, ax = plt.subplots(1,1, figsize=(15, 5))

    # Set the figure title, and the axes labels.
    the_title = fig.text(0.25, 1, 
                         'Concentrations Results from t = %.3fs to %.3fs\n Courant Number = %.1f\n approximated polynomial = %.1f\namplitude error between first and last timestep = %.7f' 
                         % (0, dt*timesteps,cr,order,error))
    ax.set_ylabel('Concentration')
    ax.set_xlabel('Grid Point')

    # We use color to differentiate lines at different times.  Set up the color map
    cmap = plt.get_cmap('copper_r')
    cNorm  = colors.Normalize(vmin=0, vmax=1.*timesteps)
    cNorm_inseconds = colors.Normalize(vmin=0, vmax=1.*timesteps*dt)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # Only try to plot 20 lines, so choose an interval if more than that (i.e. plot every interval lines)
    plotsteps = (np.arange(0, timesteps, timesteps/20) + timesteps/20).astype(int)
    #plotsteps = cmatrix.shape[0]
    ax.plot(cmatrix[0, :], color='r', linewidth=3)
    # Do the main plot
    for time in  plotsteps:
        colorVal = scalarMap.to_rgba(time)
        ax.plot(cmatrix[time, :], color=colorVal)
    # Add the custom colorbar
    ax2 = fig.add_axes([0.95, 0.05, 0.05, 0.9])
    cb1 = colorbar.ColorbarBase(ax2, cmap=cmap, norm=cNorm_inseconds)
    cb1.set_label('Time (s)')
    return


def advection(timesteps,cr):
    ''' Entry point for the Bott Scheme'''
    order = 4
    dx, u, dt, Numpoints, epsilon, cmatrix = initialize(timesteps, cr)
    ltable = gettable(order)
    cmatrix = step_advect(timesteps, ltable, cmatrix, order, Numpoints, u, dt, dx, epsilon)
    make_graph(cmatrix, timesteps, Numpoints, dt, cr, order)
    error = np.max(cmatrix[0,:]) - np.max(cmatrix[-1,:])
    return cmatrix, error

# def main():
#     #advection(60,lab_example=False)
#     advection3(60,3,lab_example=False)
# if __name__ =='__main__':
#     main()
