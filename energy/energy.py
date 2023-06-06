#Energy module using tensors (no autograd)

import numpy as np
import parameters
import torch

#Paper 1 = Kellaris
#Paper 2 = Yeh (Port Hamiltonian)
#Geometry is from Paper 1
#Energy is a mix of paper 1 & 2, and some simplifications from paper 2

#Function to get damping from geometric variables and parameters of the system (k)
#Returns b & Fd (damping?)
#Currently not used due to sys id
""" def get_damping(l__h, l__v, d_lh_dq, d_lv_dq, qdot,k_id):
	b1h = k_id[4]
	b0h = k_id[5]
	
	b1v = k_id[8]
	b0v = k_id[9]
	
	bv_q = b1v * l__v + b0v
	bh_q = b1h * l__h + b0h
	
	b = bh_q * d_lh_dq + bv_q* d_lv_dq
	Fd = b*qdot

	return b, Fd """

#Takes Q (Charge) and C_alpha (total capacitance) as input, and returns dHc/dQ
#Called every iteration
def get_dHc_dQ(Q,C_alpha):
	return Q / (C_alpha)

#Takes Q (Charge), C_alpha and dC_alpha/d_alpha and returns dHc/dalpha
#Called every iteration
def get_dHcdalpha(Q,C_alpha, d_C_alpha_d_alpha):
	return (-0.5 * Q**2 * (1/C_alpha**2)) * d_C_alpha_d_alpha #Applying chain rule to get dHc/dalpha from dHc/dC_alpha * dC_alpha/dalpha

#Takes p (momentum), m (mass), dq/dalpha, and returns dHp/dp
#Called every iteration
def get_dHpdp(p,m,d_q_d_alpha):
	#Hp = p^2/2m, dHp/p = p/m. Because we parametrize q = f(alpha), divide by dq/dalpha
	return ((p/m)/d_q_d_alpha)

#Takes dq/dalpha, q (displacement) and k, and returns dHs/dalpha
#Called every iteration
def get_dHsdalpha(d_q_d_alpha,q,k):
	return k*q*d_q_d_alpha #Chain rule: Hs = 1/2 kq^2 -> dHs/dq = kq -> dHs/dalpha = kq * dq/dalpha

#Get the energy of the system for the spring component (Hs), from the spring coefficient and displacement q
#Unlike Paper 2, we are only assuming a uni-axial spring system (not biaxial for different coefficients in vertical & horizontal directions)
def get_Hs(k, q):
	return (0.5 * k * q**2) #Spring equation (Eq. 6 in paper 2)

#From m (mass), dq/dalpha, get dHg/dalpha, using 'g' (gravity) from parameters. Hg is energy of system from potential energy (gravity)
#Called every iteration
def get_dHgdalpha(m,d_q_d_alpha):
	return m*parameters.param['g']*d_q_d_alpha #Chain rule applied to Hg = mgq

#Get the electrical contribution of the energy of the system, from Q (charge) and geometric parameters of the system
#Return Hc, the electrical energy of the system, C_alpha, the capacitance of the system, and d_C_alpha_d_alpha, the derivative of capacitance wrt alpha (the parameter)
#Called every iteration
def get_Hc(Q,le, d_le_d_alpha,alpha):
	param = parameters.param
	w =  param['w']
	t = param['t']
	eps_0 = param['eps_0']
	eps_r = param['eps_r']
	gamma = eps_0 * eps_r #Relative permittivity * Permittivity of free space
	Le = param['Le']
	
	C_alpha = (gamma * parameters.param['w'] * le) / (2*t) #Capacitance of system (Eq. 8 in paper 1). le is a function of alpha
	#For now, assuming no contribution to capacitance in non-zipped region
	# C_np = 0.0
	# C_alpha = C_p + C_np #Total capacitance of system, as a function of alpha (since le depends on alpha)
	# Hc = (0.5*Q**2)/(C_alpha) #Electrical free energy of system (Eq. 9 in paper 1)

	#Derivatives
	d_C_alpha_d_alpha_p = ((gamma * w) / (2*t)) * d_le_d_alpha #Chain rule applied to capacitance for zipped part
	
	d_C_alpha_d_alpha_np = (gamma * w * torch.cos(alpha) * (Le - le) / (0.2e1 * torch.sin(alpha) * (Le - le) + (2 * t))) #Chain rule applied to capacitance for non-zipped part
	d_C_alpha_d_alpha = d_C_alpha_d_alpha_p + d_C_alpha_d_alpha_np
	
	return C_alpha, d_C_alpha_d_alpha

#Using all the above functions, get the co-energy matrix of the system. IE, how the total energy H changes wrt alpha, for different components (electrical, spring, kinetic, potential)
#Called every iteration
def get_dHdx(p,Q,le,m, d_le_d_alpha,d_q_d_alpha, alpha,k,q):
	#Electrical contribution
	C_alpha, d_C_alpha_d_alpha = get_Hc(Q,le, d_le_d_alpha,alpha)
	dHcdalpha = get_dHcdalpha(Q,C_alpha, d_C_alpha_d_alpha)
	dHcdQ = get_dHc_dQ(Q,C_alpha)
	
	#Spring contribution
	dHsdalpha = get_dHsdalpha(d_q_d_alpha,q,k)
	
	#Kinetic contribution
	dHpdp = get_dHpdp(p,m,d_q_d_alpha)
	
	#Potential contribution
	dHgdalpha= get_dHgdalpha(m,d_q_d_alpha)
	
	#Summing up 
	dHdq = dHcdalpha + dHgdalpha+dHsdalpha #The electrical energy (Hc), potential energy (Hg) and spring energy (Hs) vary wrt q (and thus alpha). Contribution of displacement (and thus alpha) to energy.
	dHdp = dHpdp #The kinetic energy (Hp) varies wrt p (which varies wrt v, which varies wrt q and thus alpha). Contribution of momentum to energy.
	dHdQ = dHcdQ #Since Q only contributes to the total electrical energy of the system (Hc) thus dHdQ = dHcdQ. Contribution of charge Q to energy
	# print(f"TYPE OF dHdq: {type(dHdq)}") #Debug
	# print(f"TYPE OF dHdp: {type(dHdp)}")
	# print(f"TYPE OF dHdQ: {type(dHdQ)}")
	dHdx = torch.vstack((dHdq, dHdp, dHdQ))
	#Assembling the co-energy variable matrix: paper 2, eq 13
	return dHdx

#Get the matrix JR from r, b, and dq/dalpha
def get_JR(r,b,d_q_d_alpha):
	J = torch.tensor(
		[[0, 1, 0], 
		[-1, 0, 0], 
		[0, 0, 0]])
	
	R = torch.tensor(
		[[0, 0, 0],
		[0, b/d_q_d_alpha, 0],
		[0, 0, 1/r]]) #From equation 14 in paper 2. B is damping, since we have parametrized wrt alpha, use chain rule to get derivative wrt q instead of alpha
	
	J_R = J-R
	return J_R