#Module to define parameters for system

#Considering the entire 14 pouches
n_pouches = 14
Lp = 0.00986*n_pouches #Total length of pouches
Le = 0.00494*n_pouches #Total length of electrodes
w = 0.09025 #Width of each pouch
t = 15e-6 #Thickness of each pouch
eps_0 = 8.85e-12 #Permittivity of free space (Eo) used for capacitance
eps_r = 3.15 #Relative permittivity of dielectric (also dielectric constant K) used for capacitance
g = 9.81 #Gravity
m_hasel = 0.0149 #Weight of hasels
m = 0.155 #Load

param = {'g': g, 'm_hasel': m_hasel, 'm':m, 'n_pouches':n_pouches, 'Lp': Lp, 'Le': Le, 'w': w, 't': t, 'eps_0':eps_0, 'eps_r':eps_r}