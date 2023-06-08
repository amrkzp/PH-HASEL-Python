""" Open loop simulation without controller

1. Input: Sine Wave of voltage, amplitude: 6000V, bias = 3000 (the constant DC that is the baseline underneath the varying sine wave),
frequency = 0.1Hz, angular frequency = 2pif = 0.62832 radians/s. Everything else 0 or default
2. V = 6000
3. Inputs to dHdX finder:
    1. Alpha0 (Initial angle alpha)
    2. k = 458.18 (spring constant)
    3. b = 0.00031737 (damping)
    4. r = 55600000 (resistance)
    5. alpha_max (max angle of alpha)
    6. Lp (Length of pouch)
    7. A (area of pouch)
    8. Lv (total initial height/length of actuator)
    9. m = 0.155
    10. Q (charge): how/where to get?
    11. p (momentum): how/where to get?
    12. alpha : how/where to get?
4. Using those inputs, calculate dHdX from function
    Output:
    1. alpha_dot: derivative of angle alpha
    2. p_dot: derivative of momentum
    3. Q_dot: derivative of charge Q
    4. le
    5. lp
    6. q (along with time data in a time series)
4. Take voltage input from DAQ: u
    - Multiply it by 1/r. So we do V / r = I (current)
    - Add Current (I) to Qdot (derivative of system (H) wrt Q) (Why do we do this?)
    - Integrate wrt to Qdot to get Q (output)
5. Integrate p_dot wrt to dp to get p (momentum)
6. Integrate alpha_dot wrt to dAlpha to get alpha (angle)
7. Output q, p, Q every iteration (and feed it back into the simulation) [q is amplified by 1000]

Question: where do initial values for Q, p and alpha come from? [Alpha initial can be alpha0]
2. Why is p, q and pdot not moving from 0?
3. How do we do the integration, where does the time signal come into play?

"""

#------------------------------------------------------------------------------------------------------------------------#
import time
import geometry.geometry3 as geometry #CHANGE to geometry.geometry for tensors
import energy.energy2 as energy
import numpy as np
import parameters
import matplotlib.pyplot as plt
import sys
import math
# import torch #UNCOMMENT for tensors

#Tools for debugging memory & time performance
# import os
# import psutil
# from memory_profiler import profile
# import gc #For cleanup

#Returns alpha_dot, p_dot, Q_dot, le, lp, q
def update_open_loop_state(alpha, p, Q, Lv, Lp, A, alpha_max, alpha0, m, r, b, k, debug):
    #First check if alpha is out of the permissible range, if so, correct it
    if alpha.item() > alpha_max.item():
        # print("FIRED")
        alpha = np.copy(alpha_max)
        
    elif alpha.item() < alpha0.item():
        # print("FIRED")
        alpha = np.copy(alpha0)

    #Get lp (length of unzipped region of pouch), le (length of zipped region of pouch), q (displacement) and their derivatives wrt alpha 
    lp = geometry.alpha_to_lp(A,alpha)
    # print(f"lp: {type(lp)}") #Debug

    # le, d_le_d_alpha = geometry.alpha_to_le(Lp, A,alpha) #Uncomment and replace below if using tensors
    le, d_le_d_alpha = geometry.alpha_to_le(Lp, A,alpha, lp)

    #Both le and d_le_d_alpha are numpy arrays
    # print(type(le), type(d_le_d_alpha)) #Debug
    q, d_q_d_alpha = geometry.alpha_to_q(alpha, A, Lp, Lv)
    
    #Both q and d_q_d_alpha are numpy arrays
    # print(type(q), type(d_q_d_alpha)) #Debug

    #Get matrix J-R
    J_R = energy.get_JR(r,b,d_q_d_alpha)
    
    #Get the co-energy variable matrix (that contains dHdq (really dH/dalpha) for Hc, Hs and Hg, dHp/dp, and dHc/dQ)
    dH_dx = energy.get_dHdx(p,Q, le,m, d_le_d_alpha,d_q_d_alpha, alpha,k,q)

    del d_le_d_alpha
    
    # if debug:
    #     print(f"dHdq: {dH_dx[0].item()}, dHdp: {dH_dx[1].item()}, dHdQ: {dH_dx[2].item()}")
    
    #Calculate the final dHdx, derivatives of the system H (energy) wrt to the state variables. Defined in paper 2, equation (14)
    # x_dot = torch.matmul(J_R, dH_dx) #Uncomment and replace below line if using tensors
    x_dot = np.dot(J_R, dH_dx)

    del J_R
    del dH_dx

    # if debug:
    #     print(f"x_dot: {x_dot}")
    
    #Extract the individual derivatives from the matrix
    alpha_dot = x_dot[0]
    
    if alpha <= alpha0+1e-3 and alpha_dot<0.0:
        # print(f"alpha: {alpha.item()}, alpha0: {alpha0.item()}, combo: {alpha0+1e-3}")
        alpha_dot = 0.0; #tunable
        
    p_dot = x_dot[1]
    Q_dot = x_dot[2]

    del x_dot

    del le
    del lp

    # return alpha_dot, p_dot, Q_dot, le, lp, q # If we want to collect more data from model
    return alpha_dot, p_dot, Q_dot, q

# @profile
# Run open loop model with input voltage V
# Note: no matter what sample time is set, we will always bring the resulting q_out up to 0.0001s sample time to regularize all our data (since reference data / MATLAB sim is in 0.1ms)
def run_open_loop(total_time, time_step, V, alpha, p, Q, Lv, Lp, A, alpha_max, alpha0, m, r, b, k):
    num_iterations = round(total_time / time_step)
    # print(f"V = {V}") #Debug
    q_out = np.zeros(num_iterations)

    #Optional arrays to track more data:
    # alpha_dot_out = np.zeros(num_iterations)
    # p_dot_out = np.zeros(num_iterations)
    # Q_dot_out = np.zeros(num_iterations)
    
    # p_out = np.zeros(num_iterations)
    # Q_out = np.zeros(num_iterations)
    # alpha_out = np.zeros(num_iterations)

    debug = False

    # print(f"alpha0: {alpha0}") #Debug
    
    for i in range(num_iterations):
        alpha_dot, p_dot, Q_dot, q = update_open_loop_state(alpha, p, Q, Lv, Lp, A, alpha_max, alpha0, m, r, b, k, debug)

        q_out[i] = q

        # alpha.data = alpha.data + alpha_dot * time_step #Uncomment and replace below line if using tensors
        #alpha is np array, alpha_dot is float
        alpha = alpha + alpha_dot * time_step
        p += p_dot * time_step
        Q += (Q_dot + V[i]/r) * time_step
        
        #Optionally tracking more data:
        # alpha_dot_out[i] = alpha_dot
        # p_dot_out[i] = p_dot
        # Q_dot_out[i] = Q_dot

        # p_out[i] = p
        # Q_out[i] = Q
        # alpha_out[i] = alpha

        # if i % 10000 == 0:
        #     # print(h.heap()) #Heapy statistics
        #     print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) #PS Util statistics

        del alpha_dot
        del p_dot
        del Q_dot
        del q
        # gc.collect() #Forcing garbage collector for mem cleanup

    del V
    repeat = round(time_step/0.0001)
    #Bringing all the data to 0.0001s time step (by repeating/hold method (not linear interpolation), if sample time was lower) 
    if (repeat != 1):
        q_out = np.repeat(q_out, repeat)
    q_out *= 1000 #Since MATLAB model and reference data add 1000 gain
    return q_out #, alpha_dot_out, p_dot_out, Q_dot_out, p_out, Q_out, alpha_out #optionally collecting more data


#____________________Testing_____________________________________#

def generate_sine_chirp(total_time, time_step, freq, bias, amp):
    time = np.arange(0, total_time, time_step) #Starts from t = 0
    V = bias + amp * np.sin(2 * np.pi * freq * time)
    return V

#Test model with a sine chirp
def test(sim_time, time_step, V, output_file):
    #Measuring script running time
    start = time.time()
    
    alpha_0 = geometry.get_alpha0() #NUMPY float64
    # alpha_init = torch.tensor(alpha_0+1e-4, dtype=float, requires_grad=True) #Uncomment these 2 lines and replace below lines if using tensors
    # alpha_0 = torch.clone(alpha_init) #CHANGE FOR TENSORS
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

    # alpha_dot, p_dot, Q_dot, le, lp, q = update_open_loop_state(alpha=alpha_init, p=p_init, Q=Q_init, Lv=Lv, Lp=Lp, A=A, alpha_max=alpha_max, alpha0=alpha_init, m=m, r=r, b=b, k=k)

    q_out = run_open_loop(total_time=sim_time, time_step=time_step, V=V, alpha=alpha_init, p=p_init, Q=Q_init, Lv=Lv, Lp=Lp, A=A, alpha_max=alpha_max, alpha0=alpha_0, m=m, r=params['r'], b=params['b'], k=params['k'])
        
    print(f"q:{q_out[-1]}")

    np.savetxt('./Data/' + output_file, q_out, delimiter=',') #Exporting data to file

    end = time.time()
    print(end - start)
    return q_out

#Plot output of a simulation compared to reference data
def plot_simulation_output(sim_data, sim_time, time_step):
    time_data = np.arange(0, sim_time, time_step) #Collected over 20s period, with 0.0001 sample size
    fig, axs = plt.subplots(1)
    axs.plot(time_data, sim_data, label='Model Simulation')
    axs.set(xlabel='Time', ylabel='Actuator Stroke')
    axs.legend()
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    if (len(args) < 2):
            print("Usage: open_loop.py -u OR \nopen_loop.py [sim_time] [time_step] [test_method] [output_file] [debug]")
            sys.exit()

    if str(args[1]) == '-u':
        #Interactive User Input enabled
        sim_time = int(input("Simulation time (int): "))
        time_step = float(input("Sample time step: "))
        if (not math.isclose(0, (sim_time / time_step) - int(sim_time / time_step))):
                print("Sim time and sample time do not match, exiting...")
                sys.exit()
        test_method = input("Test type: (sine or step): ")
        output_file = input("Output File Name: ") + '.txt'
        debug = input("Show plot? (y/n): ")

    else:
        #CLI
        if (len(args) < 6):
            print("Usage: open_loop.py [sim_time] [time_step] [test_method] [output_file] [debug] [/custom_data_file]")
            sys.exit() 
        else:
            sim_time = int(args[1])
            time_step = float(args[2])
            if (not math.isclose(0, (sim_time / time_step) - int(sim_time / time_step))):
                print("Sim time and sample time do not match, exiting...")
                sys.exit()
            test_method = str(args[3])
            output_file = str(args[4]) + '.txt'
            debug = str(args[5])

    #Sine test with 6000V amplitude, 3000V bias, 0.1Hz frequency (specified sample time)
    if (test_method == "sine"):
        V_amp = 6000
        bias = 3000
        freq = 0.1
        V = generate_sine_chirp(sim_time, time_step, freq, bias, V_amp) #Generates V at the specified sample time

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
            if len(args) != 7:
                print("Custom data specified, but no file provided")
                sys.exit()
            file_name = args[6]
        V = np.loadtxt("./Data/" + file_name + ".txt") #Input data must be at 0.1ms sample time

        #If our simulation sample time is > 0.1ms, we need to scale down the input voltage for the simulation
        scale_down = round(time_step/0.0001)
        if (scale_down > 1):
            V = V[::scale_down]

    else:
        print("Invalid method")
        sys.exit()
    
    print("Running...")
    sim_data = test(sim_time, time_step, V, output_file)
    if (debug == 'y'):
        plot_simulation_output(sim_data, sim_time, 0.0001)