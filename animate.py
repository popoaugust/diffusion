import numpy as np

def animate_JV(t_vector, V_scan, J_total):
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig = plt.figure()
    ax = plt.axes(
        xlim=(np.min(V_scan), np.max(V_scan)), 
        # ylim=(-5, 25)
        )
    line, = ax.plot([], [], lw=2)
    
    # initialization function
    def init():
        # creating an empty plot/frame
        line.set_data([], [])
        return line,
    
    # lists to store x and y axis points
    xdata, ydata = [], []
    
    dt = 0.1
    values = np.arange(0, t_vector[-1], dt)
    V_scan_interp = np.interp(values, t_vector, V_scan)
    J_total_interp = np.interp(values, t_vector, J_total)
    
    ax.set_ylim([np.min(J_total_interp), np.max(J_total_interp)])
    
    def animate(i):
        x = V_scan_interp[i]
        y = J_total_interp[i]
        
        xdata.append(x)
        ydata.append(y)
        line.set_data(xdata, ydata)
        
        return line,
   
    # setting a title for the plot
    plt.title('J-V Scan Animation')
    # hiding the axis details
    plt.axis('on')
    
    # call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len
        (J_total_interp), interval=55, blit=True)
    
    # save the animation as mp4 video file
    anim.save('JV.gif',writer='pillow')
    
    return anim

def animate_distribution(t_vector, x_mesh, particle, title):
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig = plt.figure()
    ax = plt.axes(xlim=(np.min(x_mesh), np.max(x_mesh)), ylim=(np.min(
        particle), np.max(particle)))
    line, = ax.plot([], [], lw=1)
    
    # initialization function
    def init():
        # creating an empty plot/frame
        line.set_data([], [])
        return line,
    
    # lists to store x and y axis points
    xdata, ydata = [], []
    
    dt = 0.1
    values = np.arange(0, t_vector[-1], dt)
    
    particle_interp = np.zeros((x_mesh.size, values.size))
    
    for i in range(0, x_mesh.size):
        particle_interp[i,:] = np.interp(values, t_vector, particle[i,:])
    
    def animate(i):
        x = np.copy(x_mesh); x[0] = None
        y = particle_interp[:,i]
        xdata.append(x); xdata[i-1] = np.zeros(x_mesh.size)
        ydata.append(y); ydata[i-1] = np.zeros(particle_interp[:,i].size)
        
        line.set_data(xdata, ydata); plt.title(title); time = int(values[i
            -1])+1; plt.title('%i s' %time, loc='right')
        return line,
        
    # hiding the axis details
    plt.axis('on')
    
    frames=len(values)
    # call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=
        frames, interval=55, blit=True)
    
    # save the animation as mp4 video file
    anim.save(f'{title}.gif', fps=frames/t_vector[-1] , writer='pillow'
        )
    
    return anim