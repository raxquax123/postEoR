H_0 = 68.73
little_h = H_0 / 100
omega_b = 0.046
omega_m = 0.29
omega_lambda = 1 - omega_m # assumes flat cosmology and negligible radiation energy density
hydrogen_baryon_frac = 0.75
c = float(299792458)
k_B = 1.38e-23
h = 6.63 * 10**(-34)
T_CMB = 2.73
G = 6.67*10**(-11)
A_10 = 2.86888e-15 # einstein coefficient for hi spin-flip
m_H = 1.6735e-27 # mass of hydrogen atom
nu_21 = 1420.405751768 * 10**6


# instrument specs
A_e = 1 # effective receiver area
t_0 = 1 # total observing time
n_baselines = 1 # number density of baselines
T_rcv = 1 # receiver temperature - for meerkat, mean is 9.8K
delta_z=0.001 # smoot and debono 2017
zenith_angle = 0 # angle between zenith and observation region 
tau_0 = 1 # atmospheric zenith opacity
T_ground = 280 # ground temperature in kelvin
N_beam = 1
N_pol = 2