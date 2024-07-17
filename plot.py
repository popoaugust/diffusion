import matplotlib.pyplot as plt
import numpy as np

def plot_distributions_slm(x, u, params):
    N = params.N
    
    P = u[0:N+1,-1]
    phi = u[N+1:2*N+2,-1]
    n = u[2*N+2:3*N+3,-1]
    p = u[3*N+3:4*N+4,-1]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(x, P, label='ions')
    axes[0].set_xlabel('non-dimensional space')
    axes[0].set_ylabel('non-dimensional P')
    axes[0].set_title("Ion Distribution")
    axes[0].legend()

    axes[1].plot(x, phi, label='phi')
    axes[1].set_xlabel('non-dimensional space')
    axes[1].set_ylabel('non-dimensional phi')
    axes[1].set_title("Electric Potential")
    axes[1].legend()
    
    axes[2].plot(x, n, label='n')
    axes[2].set_xlabel('non-dimensional space')
    axes[2].set_ylabel('non-dimensional n')
    axes[2].set_title("Electron Distribution")
    axes[2].legend()
    
    axes[3].plot(x, p, label='p')
    axes[3].set_xlabel('non-dimensional space')
    axes[3].set_ylabel('non-dimensional p')
    axes[3].set_title("Hole Distribution")
    axes[3].legend()
    
    return fig, axes

def plot_distributions_tlm(x, xE, xH, u, params):
    N = params.N
    NE = params.NE
    NH = params.NH
    
    P = u[0:N+1,-1]
    phi = u[N+1:2*N+2,-1]
    n = u[2*N+2:3*N+3,-1]
    p = u[3*N+3:4*N+4,-1]
    phiE = u[4*N+4:4*N+NE+4,-1]
    nE = u[4*N+NE+4:4*N+2*NE+4,-1]
    phiH = u[4*N+2*NE+4:4*N+2*NE+NH+4,-1]
    pH = u[4*N+2*NE+NH+4:4*N+2*NE+2*NH+4,-1]
    
    xtotal = np.concatenate((-xE[::-1], x, xH), 0)
    phitotal = np.concatenate((phiE, phi, phiH), 0)
    ntotal = np.concatenate((nE, n, n[-1]*np.ones(NH)))
    ptotal = np.concatenate((p[0]*np.ones(NE), p, pH))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(x, P, label='ions')
    axes[0].set_xlabel('non-dimensional space')
    axes[0].set_ylabel('non-dimensional P')
    axes[0].set_title("Ion Distribution")
    axes[0].legend()
    
    axes[1].plot(xtotal, phitotal, label='phi')
    axes[1].set_xlabel('non-dimensional space')
    axes[1].set_ylabel('non-dimensional phi')
    axes[1].set_title("Electric Potential")
    axes[1].legend()
    
    axes[2].plot(xtotal, ntotal, label='n')
    axes[2].set_xlabel('non-dimensional space')
    axes[2].set_ylabel('non-dimensional n')
    axes[2].set_title("Electron Distribution")
    axes[2].legend()
    
    axes[3].plot(xtotal, ptotal, label='p')
    axes[3].set_xlabel('non-dimensional space')
    axes[3].set_ylabel('non-dimensional p')
    axes[3].set_title("Hole Distribution")
    axes[3].legend()
    
    return fig, axes

def plot_JV(psi, J, *dim, params):
    fig, ax = plt.subplots()
    ax.plot(psi, J, label='J')
    
    voltage_range = [params.Vi, params.Vf]
    upper_lim = np.max(voltage_range)
    lower_lim = np.min(voltage_range)
    
    if ('non-dimensional' in dim):
        ax.set_xlabel('psi - non-dimensional applied voltage')
        ax.set_ylabel('J - non-dimensional current-density')
        ax.set_xlim([lower_lim/params.Vt, upper_lim/params.Vt])
        # ax.set_ylim([0,1])
    elif ('dimensional' in dim):
        ax.set_xlabel('Applied Voltage (V)')
        ax.set_ylabel('Current Density ($mA \cdot cmˆ{-2}$)')
        ax.set_xlim([lower_lim, upper_lim])
        # ax.set_ylim([-5,25])
    
    ax.set_title("J-V Scan")
    ax.legend()
    
    ax.plot([-100,100], [0,0], color='black', linestyle='dashed',
        linewidth=0.5)
    
    return fig, ax



