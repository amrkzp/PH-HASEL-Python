#Script to compare MATLAB continuous model, MATLAB discrete model, and Python model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

ref_time_step = 0.0001 #The sample time collected from the matlab model 

def compare(ref_file, model_file, label1, label2, tolerance, ref_sim_time, ref_time_step):
    ref_data = np.loadtxt('./Data/' + ref_file) #Reference data

    model_data = np.loadtxt('./Data/' + model_file) #Python data

    ref_length = ref_data.shape[0]
    model_length = model_data.shape[0]
    if (ref_length > model_length):
        interpolate_factor = round(ref_length/model_length)
        model_data = np.repeat(model_data, interpolate_factor)

    elif(model_length > ref_length):
        shrink_factor = round(model_length/ref_length)
        model_data = model_data[::shrink_factor]

    time = np.arange(0, ref_sim_time, ref_time_step)

    tol = tolerance * model_data
    error = model_data - ref_data

    NRMSE = np.sqrt(((error)**2).mean()) / ref_data.mean()

    fig, axs = plt.subplots(2)
    fig.suptitle("Reference vs Model (Open Loop)")
    axs[0].plot(time, ref_data, label=label1) 
    axs[0].plot(time, model_data, label=label2)
    axs[0].set(xlabel='Time', ylabel='Actuator Stroke')
    at = offsetbox.AnchoredText(f"Fit: {(1-NRMSE)*100:.2f}%",
                      loc='lower left', prop=dict(size=8), frameon=True, bbox_to_anchor=(0., 1.),
                       bbox_transform=axs[0].transAxes
                      )
    axs[0].add_artist(at)

    axs[1].plot(time, tol, 'g', label=f"Tolerance {tolerance}")
    axs[1].plot(time, -tol, 'g')
    axs[1].plot(time, error, 'r', label='Error')
    axs[1].set(xlabel='Time', ylabel='Stroke Error')
    axs[1].set_title("Error")

    plt.fill_between(time, tol, -tol, color='green', alpha=0.2)
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    plt.show()

    print(f"Max absolute error: {np.max(error)}")

if __name__ == '__main__':
    #Note: can replace UI with command line args for script
    ref_file = input('Ref data file name: ') + '.txt'
    model_file = input('Model data file name: ') + '.txt'
    ref_name = input('Ref data plot name: ')
    model_name = input('Model data plot name: ')
    ref_exp_time = int(input('Ref experiment time (int): '))
    ref_exp_sample_time = float(input('Ref experiment sample time: '))
    tol = float(input('Tolerance: '))
    compare(ref_file, model_file, ref_name, model_name, tol, ref_exp_time, ref_exp_sample_time)