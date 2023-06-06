#Geometry module using tensors and autograd

import numpy as np
import parameters
import scipy.optimize
import torch

#Function for the Area of the pouch, as defined in Paper 1 (Equation 1)
def area(x, A, Lp):
    return (0.5 * Lp**2 * (x - (np.sin(x)*np.cos(x)))/x**2 - A)

#Step 1: Get the initial angle alpha0, from the area.
#This function is only called once to initialize alpha
def get_alpha0():
    Lp = parameters.param['Lp']
    #Assume that the area is constant (since fluid is incompressible). 
    # Thus, the area is equivalent to the area of the circle formed at maximum deformation.
    # Assuming the electrodes are fully zipped at max deformation (IE, le = Le),
    # The circumference of the circle C is 2*(Lp - Le). Since C = 2(pi)r, r = C/2(pi). A = (pi)r^2. Thus A = C^2/4(pi) = (Lp-Le)^2/(pi)
    A = (Lp-parameters.param['Le'])**2 / np.pi

    #Given A and Lp, use eq 1 from paper 1 to solve for alpha 0
    a0 = scipy.optimize.fsolve(area, 1, (A, Lp))[0]
    del A
    del Lp
    return a0

#Step 2: get the initial length / height of the actuator (referred to as h in the paper, Lv in patricia's code): Eqn 2 in paper 1
#This function is only called once to initialize height h = Lv
def get_initial_length():
    a0 = get_alpha0()
    return parameters.param['Lp'] * (np.sin(a0)/a0)

#Step 3: calculate the max possible value of alpha, given Le, Lp. Assuming condition 1 in paper 1, IE Le <= lmax.
#Finding max length of zipped part of electrodes, assuming electrodes are fully zipped: le = Le
#le = Lp - lp becomes Le = Lp - lp. lp = Lp - Le. Use equation (3) from paper 1 and substitute with lp
#This function is only called once, to get alpha_max
def GetAlphaMaxByArea(A, Le, Lp):
    return (scipy.optimize.fsolve(max_alpha, 1, (A, Le, Lp))[0])

#Function to solve for max alpha, derived from eqn (3) & (4) from paper 1
def max_alpha(x, A, Le, Lp):
    return (np.sqrt((2*A*x**2)/(x-np.sin(x)*np.cos(x))) + Le - Lp)

#Step 4: get lp(alpha) from alpha (and constant A), using equation 3 in paper 1. Also calculate the derivative
#Alpha input alpha needs to be a torch tensor scalar
#This function is called TWICE PER every iteration (for each sample)
def alpha_to_lp(A, alpha):
    lp = torch.sqrt((2*A*alpha**2)/(alpha - torch.sin(alpha) * torch.cos(alpha)))
    lp.backward(retain_graph=True)
    # print(f"WITHIN d_lp_d_alpha: {alpha.grad}") #Debug
    d_lp_d_alpha = alpha.grad.data
    alpha.grad.data.zero_()
    
    return lp, d_lp_d_alpha

#Step 5: get le(alpha) from Lp and lp(alpha) using eqn 4 in paper 1
#This function is called every iteration (for each sample)
def alpha_to_le(Lp, A, alpha):
    lp = alpha_to_lp(A, alpha)[0]
    # print(f"A: {A}, alpha: {alpha.item()}, lp: {lp}") #Debug
    le = Lp - lp
    
    le.backward(retain_graph=True)
    # print(f"WITHIN d_le_d_alpha: {alpha.grad}") #Debug
    d_le_d_alpha = alpha.grad.clone()
    alpha.grad.data.zero_()
    del lp
    return le, d_le_d_alpha

#Step 6: Get displacement (q) from alpha, A, Lp and h. Use eqn 5 from paper 1
# Called every iteration (for each sample)
def alpha_to_q(alpha, A, Lp, h):
    lp = alpha_to_lp(A, alpha)[0] #CAN WE REMOVE THIS?
    lp.retain_grad()
    # print(f"lp: {lp}, d_lp_d_alpha: {d_lp_d_alpha}") #Debug

    le = alpha_to_le(Lp, A, alpha)[0]
    le.retain_grad()
    # print(f"le: {le}, d_le_d_alpha: {d_le_d_alpha}") #Debug
    
    q = h - (lp*(torch.sin(alpha)/alpha) + le)

    q.backward(retain_graph=True)
    d_q_d_alpha = alpha.grad.clone()
    # print(f"d_q_d_alpha: {a.grad}") #Debug
    alpha.grad.data.zero_()
    lp.grad.data.zero_()
    le.grad.data.zero_()
    
    del h
    del le
    del lp
    
    return q, d_q_d_alpha


#------------------------------------------Tests ---------------------------------------:
# Lp = parameters.param['Lp']
# Le = parameters.param['Le']
# A = (Lp-Le)**2 / np.pi
# alpha_0 = get_alpha0()
# alpha_init = torch.tensor(alpha_0, dtype=float, requires_grad=True)
# print(Le)
# print(f"A: {A}")
#print(GetAlphaMaxByArea(A, Le, Lp))
# lp, d_lp_d_alpha = alpha_to_lp(A, torch.tensor(0.241, requires_grad=True))
# print(f"lp: {lp}, dlp: {d_lp_d_alpha}")
# le, d_le_d_alpha = alpha_to_le(Lp, A, torch.tensor(0.241, requires_grad=True))
# print(f"le: {le}, dle: {d_le_d_alpha}, Lp: {Lp}")
# q, d_q_d_alpha = alpha_to_q(0.3, A, Lp)
# print(f"q: {q.item()}, d_q_d_alpha: {d_q_d_alpha.item()}")