# PH-HASEL-Python
Port Hamiltonian Open-Loop Simulation for HASELs in Python, based on Kellaris et al. (2019) and Yeh et al. (2022). Geometry is from Kellaris, Energy is a mix of paper 1 & 2, and some simplifications from paper 2.

This model has been adapted from a MATLAB model developed by Patricia Apostol at SRL, ETH Zurich. Note: the model has been discretized for Python (implemented as continuous integration in MATLAB), with a default sample time of 0.0001s (which can be changed easily in [open_loop.py](open_loop.py).

The model has different geometry and energy modules for (a) using tensors and autograd and (b) using only numpy arrays and symbolic differentiation. The modules can be swapped out interchangeably to test performance.

## Metrics
1. Accuracy comparison between MATLAB and Python model:
    Maximum error between the Python and MATLAB continuous model is < 2 * 10<sup>-7</sup>. For the most part, error is much below a 0.1% tolerance

![Python Model Accuracy.png](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Python%20Model%20Accuracy.png)
    
2. Run time & Memory usage for different approaches (20 second prediction, 0.0001s sample time = 200,000 samples):
    1. With tensors and autograd (slowest):
   
        Run time: 137.2 seconds
    
        Peak memory usage: 990 MiB

    ![Memory usage with tensors](https://github.com/MadhavL/PH-HASEL-Python/blob/main/With%20Tensors.png)
    
    2. Without tensors, with symbolic differentiation:
    
        Run time: 18.7 seconds (7X faster)
    
        Peak memory usage: 67 MiB (15X less)

    ![Memory usage without tensors](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Without%20Tensors%20With%20Symbolic.png)
    
    3. Without tensors, without symbolic differentiation:
    
        Run time: 7.5 seconds
        
        Note: Symbolic differentiation (just the computation for derivatives) adds around 11s to the simulation!

## Instructions to Run:
1. Specify system parameters in [parameters.py](parameters.py)
2. Set up simulation in [open_loop.py](open_loop.py), line 180 onwards: set voltage, frequency, length of iteration, sample time step, export file name)
3. Data will be exported to file ['modeldata.txt'](modeldata.txt)
4. Use [compare.py](compare.py) to plot data (or use custom scripts for other tasks)
