import numpy as np
import open_loop
import geometry.geometry3 as geometry
import parameters
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import time
from scipy import optimize
import sys
import math
import signal

#Globals to track number of iterations, time and loss for optimizer (not needed if we don't want to track optimization progress)
Niter = 0
time_track = []
loss_track = []
start_time = 0

def interrupt_handler(sig, frame):
    #Plot optimization graph
    fig, axs = plt.subplots(1)
    axs.plot(time_track, loss_track)
    axs.set(xlabel='Time', ylabel='MSE Loss')
    plt.yscale('log')
    plt.show()
    sys.exit()

#Calculate the mse_loss from running the specified simulation, wrt reference data
def mse_loss(sys_parameters, sim_time, time_step, V, reference_data, plot_data):
    global loss_track
    global start_time

    #Initialize the model with system geometry
    alpha_0 = geometry.get_alpha0() #NUMPY float64
    alpha_init = np.array(alpha_0+1e-4, dtype=np.float64)
    alpha_0 = np.copy(alpha_init)
    p_init = 0
    Q_init = 0
    params = parameters.param
    Lv = geometry.get_initial_length()
    Lp = params['Lp']
    Le = params['Le']
    A = (Lp-Le)**2 / np.pi
    alpha_max = np.array(geometry.GetAlphaMaxByArea(A, Le, Lp), dtype=np.float64)
    m = params['m'] + params['m_hasel']

    #Get current iteration's physical parameters which will be used in the simulation:
    k, b, r = sys_parameters

    #Run simulation
    # start = time.time() #Debug run time per iteration
    q_out = open_loop.run_open_loop(total_time=sim_time, time_step=time_step, V=V, alpha=alpha_init, p=p_init, Q=Q_init, Lv=Lv, Lp=Lp, A=A, alpha_max=alpha_max, alpha0=alpha_0, m=m, r=r, b=b, k=k)
    
    # Print simulation output & run-time for debugging
    # print(f"q_out:{q_out[-1]}")
    # end = time.time()
    # print(f"Run time: {end - start}")

    #Calculate MSE loss
    mse = ((reference_data - q_out)**2).mean()
    print(f"MSE: {mse}") # Debugging

    #Compare simulation data to reference for each iteration
    compare_simulation_output(reference_data, q_out, plot_data) #Hardcoded time step to 0.1ms since reference data and q_out will always be in 0.1ms samples

    #Each time the loss function is evaluated, track the time and the loss value to track progress of optimization
    time_track.append(time.time() - start_time)
    loss_track.append(mse)
    return mse

#Plot output of a simulation compared to reference data
def compare_simulation_output(ref_data, model_data, plot_data):
    fig, model_plot, annotation = plot_data
    ref_length = ref_data.shape[0]
    model_length = model_data.shape[0]
    if (ref_length != model_length):
        print("Reference and model dataset lengths do not match! (Are sample times same?)")
        return

    error = model_data - ref_data

    NRMSE = np.sqrt(((error)**2).mean()) / ref_data.mean()

    #Plotting
    model_plot.set_ydata(model_data)
    annotation.txt.set_text(f"Fit: {(1-NRMSE)*100:.2f}%")
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"Max absolute error: {np.max(error)}")

#Callback function for optimizer to print progress per iteration
def callbackF(Xi):
    global Niter
    global loss_track
    print(f"Iteration: {Niter}, k: {Xi[0]}, b: {Xi[1]}, r: {Xi[2]}, loss: {loss_track[-1]}")
    Niter += 1

def sys_id(sim_time, time_step, reference_data, V, max_evals, debug):
    global start_time

    plot_data = ()
    if (debug):
        #If plotting enabled
        fig, axs = plt.subplots(1)
        fig.suptitle("Reference vs Model (Open Loop)")
        ref_length = reference_data.shape[0]
        total_time = round(ref_length * time_step)
        time_data = np.arange(0, total_time, time_step) #Since all our data is at 0.1ms time step

        #Plotting
        axs.plot(time_data, reference_data, label='Reference') 
        model_plot = axs.plot(time_data, reference_data, label='Model')[0]
        axs.set(xlabel='Time', ylabel='Actuator Stroke')
        at = offsetbox.AnchoredText(f"Fit: {(1)*100:.2f}%",
                        loc='lower left', prop=dict(size=8), frameon=True, bbox_to_anchor=(0., 1.),
                        bbox_transform=axs.transAxes
                        )
        axs.add_artist(at)
        axs.legend()
        fig.tight_layout()
        plt.show(block=False)
        plot_data = (fig, model_plot, at)

    #Initialize sys id and set experiment time
    sys_params_init = [5e+02, 1e-04, 5e+08] #Initializing at the average of the bounds
    param_bounds = [(1e+02, 1e+03), (1e-05, 1e-03), (1e+07, 1e+10)]

    #Run the optimizer
    start_time = time.time()
    result = optimize.minimize(mse_loss, sys_params_init, 
                               (sim_time, time_step, V, reference_data, plot_data), 
                               method='Nelder-Mead', 
                               bounds=param_bounds,
                               options={'maxfev':max_evals, 'disp':True}, 
                               callback=callbackF)
    print(f"Result = {result}")

    #Plot optimization graph
    fig, axs = plt.subplots(1)
    axs.plot(time_track, loss_track)
    axs.set(xlabel='Time', ylabel='MSE Loss')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    if (len(args) < 2):
            print("Usage: system_identification.py -u OR \nsystem_identification.py [sim_time] [time_step] [reference_data] [max_evals] [debug] [test_method] [/custom_input]")
            sys.exit()

    if str(args[1]) == '-u':
        #Interactive User Input enabled
        sim_time = int(input("Simulation time (int): "))
        time_step = float(input("Sample time step: "))
        if (not math.isclose(0, (sim_time / time_step) - int(sim_time / time_step))):
                print("Sim time and sample time do not match, exiting...")
                sys.exit()
        reference_data = np.loadtxt("./Data/" + input("Reference Data (0.1ms) file for sys id (no ext): ") + ".txt")
        maxevals = int(input("Max number of simulations: "))
        test_method = input("Test type: (sine, step or custom): ")
        debug = input("Show plots? (y/n): ")

    else:
        #CLI
        if (len(args) < 7):
            print("Usage: system_identification.py [sim_time] [time_step] [reference_data] [max_evals] [debug] [test_method] [/custom_input]")
            sys.exit() 
        else:
            sim_time = int(args[1])
            time_step = float(args[2])
            if (not math.isclose(0, (sim_time / time_step) - int(sim_time / time_step))):
                print("Sim time and sample time do not match, exiting...")
                sys.exit()
            reference_data = np.loadtxt("./Data/" + str(args[3]) + ".txt")
            maxevals = int(args[4])
            debug = str(args[5])
            test_method = str(args[6])

    #NOTE: Reference data should be at 0.1ms sample time!
    if debug == 'y':
        debug = True
    elif debug == 'n':
        debug = False
    else:
        print('Debug must be either y or n')
        sys.exit()

    #Get the input voltage from the test method
    #Sine test with 6000V amplitude, 3000V bias, 0.1Hz frequency (specified sample time)
    if (test_method == "sine"):
        V_amp = 6000
        bias = 3000
        freq = 0.1
        V = open_loop.generate_sine_chirp(sim_time, time_step, freq, bias, V_amp) #Generates V at the specified sample time

    #440s step voltage test (0.1ms sample time)
    elif (test_method == "step"):
        V = np.loadtxt("./Data/Step_Ref_Voltage.txt")
        #If our sample time for simulation is > 0.1ms, we need to scale down the input voltage for the simulation
        scale_down = round(time_step/0.0001)
        if (scale_down > 1):
            V = V[::scale_down]

    #Custom input data
    elif (test_method == "custom"):
        if str(args[1]) == '-u':
            file_name = input("Enter file name (without ext) for voltage data (NOTE: SHOULD BE 0.1ms sample time):")
        else:
            if len(args) != 8:
                print("Custom data specified, but no file provided")
                sys.exit()
            file_name = args[7]
        V = np.loadtxt("./Data/" + file_name + ".txt") #Input data must be at 0.1ms sample time

        #If our simulation sample time is > 0.1ms, we need to scale down the input voltage for the simulation
        scale_down = round(time_step/0.0001)
        if (scale_down > 1):
            V = V[::scale_down]
    signal.signal(signal.SIGINT, interrupt_handler)
    print("Running sys id...")
    sys_id(sim_time, time_step, reference_data, V, maxevals, debug)
    