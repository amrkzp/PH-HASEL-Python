# PH-HASEL-Python
Port Hamiltonian Open-Loop Simulation for HASELs in Python, based on Kellaris et al. (2019) and Yeh et al. (2022). Geometry is from Kellaris, Energy is a mix of paper 1 & 2, and some simplifications from paper 2.

This model has been adapted from a MATLAB model developed by Patricia Apostol at SRL, ETH Zurich. Note: the model has been discretized for Python (implemented as continuous integration in MATLAB), with a default sample time of 0.0001s (which can be changed easily in [open_loop.py](open_loop.py).

The model has different geometry and energy modules for (a) using tensors and autograd and (b) using only numpy arrays and symbolic differentiation. The modules can be swapped out interchangeably to test performance.

## Metrics
1. Accuracy comparison between MATLAB and Python model:
    Maximum error between the Python and MATLAB continuous model is < 2 * 10<sup>-7</sup>. For the most part, error is much below a 0.1% tolerance

![Python Model Accuracy.png](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Images/Python%20Model%20Accuracy.png)
    
2. Run time & Memory usage for different approaches (20 second prediction, 0.0001s sample time = 200,000 samples):
    1. With tensors and autograd (slowest):
   
        Run time: 137.2 seconds
    
        Peak memory usage: 990 MiB

    ![Memory usage with tensors](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Images/With%20Tensors.png)
    
    2. Without tensors, with symbolic differentiation:
    
        Run time: 18.7 seconds (7X faster)
    
        Peak memory usage: 67 MiB (15X less)

    ![Memory usage without tensors](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Images/Without%20Tensors%20With%20Symbolic.png)
    
    3. Without tensors, without symbolic differentiation:
    
        Run time: 7.5 seconds
        
        Note: Symbolic differentiation (just the computation for derivatives) adds around 11s to the simulation!

3. Discretization sample time:
    There is a tradeoff between accuracy & run-time: for different discretization times.

    ![Discretization Error vs Runtime](https://github.com/MadhavL/PH-HASEL-Python/blob/main/Images/Discretization%20Error.png)

    Recommended:
    * Highest accuracy: 1e-4
    * Compromise: 2e-4
    * Lower accuracy/faster: 4e-4

## Important Notes:
1. All output data (displacement) from the model is exported in a form that represents a 0.1ms sample time, even if the simulation used a different sample time, and the displacement is amplified by a factor of 1000 (since the MATLAB model and reference data use mm instead of m)
2. Simulation only accepts sample times that cleanly divide the total experiment time (eg, don't use 0.3ms sample times)
3. All data input (voltage) to the model must be input assuming a 0.1ms sample time
4. Whenever specifying input/output file names, do not add extensions ".txt" as they will automatically be added
5. All data input and output files should be in the ./Data folder
6. The fit displayed is based on NRMSE (Normalized Root Mean Squared Error)
7. Current naming convention for data files: "experiment type (sine, step)"_"source (Sim = MATLAB, Model = Python, Ref = Reference Data)"_"(Optional) Modifiers to describe modifications to data or other characteristics"_"Discretization time (tn where n is in 0.n ms discretization time)"
8. When specifying experiments, sine is a 6000V + 3000V bias sine chirp for 20s at 0.1HZ, and step is a 440s step voltage experiment. Custom means you have the input voltage data yourself (at 0.1ms sample)


## Instructions to Run:
1. Specify system parameters in [parameters.py](parameters.py)
2. Collect reference data files in the ./Data folder. All data should assume a 0.1ms sample time, so scale accordingly
3. Run [open_loop.py](open_loop.py) from the command line. Specify -u for interactive user input, otherwise use command line arguments (if you run open_loop.py without args it will tell you in what order to specify arguments)
3. Data will be exported in the ./Data folder
4. Use [compare.py](compare.py) to compare reference data with model output. Specify -u for interactive user input, otherwise use CLIs
5. Run [system_identification.py](system_identification.py) on reference data to estimate physical parameters k, b and r for the system. Specify -u for interactive input, otherwise use CLIs.
6. If you want to use tensors, make changes in [open_loop.py](open_loop.py) that have been clearly commented ("uncomment to use tensors"). WARNING - super slow!
