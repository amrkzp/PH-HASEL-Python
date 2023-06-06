#Script to compare MATLAB continuous model, MATLAB discrete model, and Python model

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.loadtxt('testdata.txt') #Matlab Discrete data
    datacontinuous = np.loadtxt('testdata_continuous.txt') #Matlab continuous data
    model = np.loadtxt('modeldata.txt') #Python data
    time = np.arange(0, 20, 1e-4) #Collected over 20s period, with 0.0001 sample size

    tolerance = 0.001

    tol = tolerance * model
    error = datacontinuous-model

    fig, axs = plt.subplots(2)
    axs[0].plot(time, data, label='Matlab Discrete') 
    axs[0].plot(time, datacontinuous, label='Matlab Continuous') 
    axs[0].plot(time, model, label='Python')
    axs[0].set_title('Open Loop Prediction Models')
    axs[0].set(xlabel='Time', ylabel='Actuator Stroke')

    axs[1].plot(time, tol, 'g', label='Tolerance (0.1%)')
    axs[1].plot(time, -tol, 'g')
    axs[1].plot(time, error, 'r', label='Error')
    # axs[1].set_title('Error (Python vs Matlab Continuous)')
    axs[1].set(xlabel='Time', ylabel='Stroke Error')

    plt.fill_between(time, tol, -tol, color='green', alpha=0.2)
    axs[0].legend()
    axs[1].legend()
    plt.show()

    print(f"Max absolute error: {np.max(error)}")