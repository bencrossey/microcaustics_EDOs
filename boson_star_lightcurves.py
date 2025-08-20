"""Script to compute and plot magnification lightcurves and caustic curves for boson stars (BS) embedded in a macrolens."""
___author___ = 'Ben Crossey'
___date___ = '2025-08-18'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import scienceplots

from scipy.interpolate import interp1d
from scipy.differentiate import derivative
from scipy.optimize import root_scalar

from multiprocessing import Pool

plt.style.use(['science'])


data = pd.read_csv('mt_boson_list.csv', header=None) #boson star mass profile
data.iloc[:, 1] = data.iloc[:, 1].astype(str).str.replace(r'\*\^', 'e', regex=True).astype(float)


#Interpolate the boson star (BS) mass profile:
def m(mtau, tau_m=1.0):
    mtau_cop = mtau.copy()
    
    # Scale tau values by tau_m
    mtau_cop.iloc[:, 0] = tau_m * mtau_cop.iloc[:, 0]

    # Extract columns
    tau_pos = mtau_cop.iloc[:, 0].values
    m_pos = mtau_cop.iloc[:, 1].values

    # Create symmetric negative tau values
    tau_neg = -tau_pos
    m_neg = m_pos

    # Combine: (tau, m), (-tau, m), and (0, 0)
    tau_all = np.concatenate([tau_neg, [0.0], tau_pos])
    m_all = np.concatenate([m_neg, [0.0], m_pos])

    # Sort by tau to ensure interp1d gets ordered input
    sorted_idx = np.argsort(tau_all)
    tau_sorted = tau_all[sorted_idx]
    m_sorted = m_all[sorted_idx]

    # Interpolation
    mtau_interp = interp1d(tau_sorted, m_sorted, kind='cubic', fill_value='extrapolate')
    return mtau_interp

def m_prime(m,tau): #derivative of mass profile with respect to tau
    return derivative(m, tau, maxiter=2, order=3)


#Defining parameters

tau_m_val = 2 # R_90/R_E for the boson star

mu_r = 3.0 #radial magnification
mu_t = 100.0 #tangential magnification

u_min = 0 #minimum impact parameter
t_E = 10 #Einstein crossing time


#Defining value ranges

time_step = 0.1
timestamps = np.arange(-50, 50, time_step) # Time range for the light curve

tau_values = np.linspace(0.001, 1000, 10000) #tau values to iterate over to find solutions to the lens equation
tau_values_CC = np.linspace(0.001, 50, 10000) #tau values to iterate over to find critical curves (and then caustics)

phi_values = np.linspace(0, 2 * np.pi, 100) #phi values to plot caustic and critiical curves


#Defining inverse magnification functions

def inverse_magnification(m,tau,phi,mu_r=mu_r, mu_t=mu_t):
    return (
        (((1/mu_r) - (m_prime(m,tau).df / (2.0 * tau)) ) * ((1/mu_t) - (m_prime(m,tau).df / (2.0 * tau)) )) -
        ((m_prime(m,tau).df / (2.0 * tau)) - (m(tau) / tau**2))**2 +
        (1/mu_r - 1/mu_t) * ((m_prime(m,tau).df / (2.0 * tau)) - (m(tau) / tau**2))*np.cos(2*phi)
    )

def inverse_magnification_vec(m_vals, m_prime_vals, tau_values, phi, mu_r=mu_r, mu_t=mu_t): #vectorized version of inverse_magnification
    #used to compute caustics
    term1 = (1/mu_r - m_prime_vals / (2.0 * tau_values))
    term2 = (1/mu_t - m_prime_vals / (2.0 * tau_values))
    term3 = (m_prime_vals / (2.0 * tau_values)) - (m_vals / tau_values**2)
    return term1 * term2 - term3**2 + (1/mu_r - 1/mu_t) * term3 * np.cos(2 * phi)


#Defining functions required to compute lightcurves

def lens_eqn(tau, m_tau, u1, u2, mu_t=mu_t, mu_r=mu_r): # Embedded boson star lens equation for radial image position(s) tau from source position (u_1, u_2)
    R_r = ((tau / mu_r) - (m_tau(tau) / tau))
    R_t = ((tau / mu_t) - (m_tau(tau) / tau))
    return (u1 / R_r)**2 + (u2 / R_t)**2 - 1  #Image positions solve this equalling zero

def phi_from_tau(tau, m_tau, u1, u2, mu_r=mu_r, mu_t=mu_t): # Calculate the azimuthal angle of the image position tau
    R_r = ((tau / mu_r) - (m_tau(tau) / tau))
    R_t = ((tau / mu_t) - (m_tau(tau) / tau))
    return np.arctan2(u2 / R_t, u1 / R_r) 

def special_solution_eqn_r(tau, m_tau, mu_r=mu_r): # Radial lens equation (for images at phi = 0, pi) for source at (u1,u2) = (0,0)
    return tau - np.sqrt(mu_r * m_tau(tau))

def special_solution_eqn_t(tau, m_tau, mu_t=mu_t): # Tangential lens equation (for images at phi = pi/2, 3pi/2) for source at (u1,u2) = (0,0)
    return tau - np.sqrt(mu_t * m_tau(tau))

def source_trajectory(t, u_min=u_min, t_E=t_E): # Source position in the source plane at time t
    u1 = u_min
    u2 = t / t_E
    return u1, u2


# Mapping lens to source coordinates
def lens_to_source_mapping(tau_m,tau,phi,mu_r=mu_r, mu_t=mu_t, data=data):
    m_tau = m(data, tau_m)
    u_1 = ((tau*np.cos(phi)) / mu_r) - ((tau*np.cos(phi))/tau**2)*(m_tau(tau))
    u_2 = ((tau*np.sin(phi)) / mu_t) - ((tau*np.sin(phi))/tau**2)*(m_tau(tau))
    return u_1, u_2


# Function to compute critical curves for a given tau_m and phi - get caustics from lens to source mapping
def compute_critical_curves(tau_m, phi, tau_values_=tau_values_CC, data=data):
    m_tau = m(data, tau_m)
    m_vals = np.array([m_tau(tau) for tau in tau_values_])
    m_prime_vals = np.array([m_prime(m_tau, tau).df for tau in tau_values_])

    im_vals = inverse_magnification_vec(m_vals, m_prime_vals, tau_values_, phi)
    signs = np.sign(im_vals)
    root_idxs = np.where(signs[:-1] * signs[1:] < 0)[0]

    roots = []
    for i in root_idxs:
        wrapper = lambda x: inverse_magnification(m_tau, x, phi)
        a, b = tau_values_[i], tau_values_[i + 1]
   
        result = root_scalar(wrapper, bracket=[a, b], method='brentq')
        if result.converged:
            roots.append(result.root)
    
    return tau_m, roots, phi


# Function to solve the BS lens equation for a given tau_m over a range of timestamps
def lens_eqn_solver(tau_m, timestamps_=timestamps,tau_values_=tau_values,data=data):

    m_tau = m(data, tau_m)

    all_solutions = []
    u1_traj = []
    u2_traj = [] # arrays to store trajectory coordinates, required for plotting later

    for t in timestamps_: 

        u1, u2 = source_trajectory(t)
        u1_traj.append(u1)
        u2_traj.append(u2)

        if np.isclose(u1, 0.0, atol=1e-6) and np.isclose(u2, 0.0, atol=1e-6):
            
            special_solutions = []

            special_sol_r = [special_solution_eqn_r(tau, m_tau) for tau in tau_values_]

            for i in range(len(tau_values_)-1):
                if special_sol_r[i] * special_sol_r[i+1] < 0:
                    a, b = tau_values_[i], tau_values_[i+1]
                    sol = root_scalar(special_solution_eqn_r, args=(m_tau,), bracket=[a, b], method='brentq')
                    if sol.converged:
                        special_solutions.append((sol.root, 0))
                        special_solutions.append((sol.root, np.pi))
            
            special_sol_t = [special_solution_eqn_t(tau, m_tau) for tau in tau_values_]

            for i in range(len(tau_values_)-1):
                if special_sol_t[i] * special_sol_t[i+1] < 0:
                    a, b = tau_values_[i], tau_values_[i+1]
                    sol = root_scalar(special_solution_eqn_t, args=(m_tau,), bracket=[a, b], method='brentq')
                    if sol.converged:
                        special_solutions.append((sol.root, np.pi/2))
                        special_solutions.append((sol.root, 3*np.pi/2))
            
            all_solutions.append(special_solutions)

        else:

            lens_eqn_vals = [lens_eqn(tau, m_tau, u1, u2) for tau in tau_values_]
            roots = []

            for i in range(len(tau_values_) - 1):
                if lens_eqn_vals[i] * lens_eqn_vals[i+1] < 0:
                    a, b = tau_values_[i], tau_values_[i+1]
                    sol = root_scalar(lens_eqn, args=(m_tau,u1,u2), bracket=[a, b], method='brentq')
                    if sol.converged:
                        root_and_phi = (sol.root, phi_from_tau(sol.root, m_tau, u1, u2))
                        roots.append(root_and_phi)

            all_solutions.append(roots) 


    return all_solutions, np.array(u1_traj), np.array(u2_traj)


# Function to calculate the lightcurve for the BS with given tau_m
def boson_star_lightcurve(tau_m,timestamps_=timestamps, tau_values_=tau_values,data=data):
    m_tau = m(data, tau_m)
    all_solutions, u1_traj, u2_traj = lens_eqn_solver(tau_m, timestamps_, tau_values_)

    magnifications = []

    for solution in all_solutions:
        magn_temp = [1/np.abs(inverse_magnification(m_tau,tau,phi)) for tau, phi in solution]
        total_magn = np.sum(magn_temp)
        magnifications.append(total_magn)
    
    return np.array(magnifications), u1_traj, u2_traj


#Plot the lightcurve and caustic curve with crossing

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Lightcurve

lightcurve, u1_traj, u2_traj = boson_star_lightcurve(tau_m_val)

ax1.plot(timestamps, lightcurve, color='black', label=r'$\tau_m$ = ' + f'{tau_m_val}')
ax1.set_xlabel(r'$t/t_E$', fontsize=22)
ax1.set_xlim(-20, 20)
ax1.set_ylabel(r'$\mu$', fontsize=22)
# ax1.set_yscale('log')
ax1.tick_params(axis='both', labelsize=20)
ax1.set_title(r'Boson Star Lightcurve with $\tau_m$ = ' + f'{tau_m_val}', fontsize=24)

# Caustic curve

arguments = [(tau_m_val, phi, tau_values_CC) for phi in phi_values]

with Pool() as pool:
    critical_curve_results = pool.starmap(compute_critical_curves, arguments)


caustic_u1 = []
caustic_u2 = []

for tau_m, roots, phi in critical_curve_results:
    for tau in roots:
        u1, u2 = lens_to_source_mapping(tau_m, tau, phi)
        caustic_u1.append(u1)
        caustic_u2.append(u2)


ax2.scatter(caustic_u1, caustic_u2, color='black', s=15)
ax2.plot(u1_traj, u2_traj, color='red', linewidth=3, label="Source Trajectory")
ax2.legend(loc='lower left', fontsize='x-large')
ax2.set_xlabel(r"$u_1$", fontsize=22)
ax2.set_ylabel(r"$u_2$", fontsize=22)
ax2.tick_params(axis='both', labelsize=20)
ax2.set_title(r"Caustic Curve for Boson Star with $\tau_m$ = " + f'{tau_m_val}', fontsize=24)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4,4)
ax2.grid(True)

plt.tight_layout()
plt.show()

