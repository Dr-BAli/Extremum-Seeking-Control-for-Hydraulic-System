import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Constants and parameters for hydraulic system (realistic values)
V_d = 9.55e-6       # Pump displacement volume (m³/rev)
omega = 1500        # Angular velocity (rpm)
k_s = 1e-12         # Slip coefficient (m³/s/Pa)
A_c = 2.5e-4        # Effective area of the piston (m²)
F_s = 100           # Static friction (N)
F_c = 50            # Coulomb friction (N)
v_s = 0.01          # Stribeck velocity (m/s)
m = 150             # Mass (kg)
c = 300             # Damping coefficient (Ns/m)
k = 20000           # Stiffness (N/m)
K_v = 5e-5          # Valve flow coefficient (m²)
n = 0.5             # Non-linearity exponent
C_d = 0.62          # Discharge coefficient
u = 0.7             # Control signal (dimensionless)
L = 15              # Length of the pipe (m)
D = 0.05            # Diameter of the pipe (m)
rho = 850           # Fluid density (kg/m³)
P_loss = 500        # Power loss due to inefficiencies (W)
h = 100             # Heat transfer coefficient (W/m²·K)
A = 2               # Surface area for heat exchange (m²)
T_env = 25          # Environmental temperature (°C)
c_p = 2000          # Specific heat capacity (J/kg·K)
eta_0 = 0.1         # Reference viscosity (Pa·s)
T_0 = 25            # Reference temperature (°C)
beta = 0.02         # Temperature coefficient (1/°C)
alpha = 0.0007      # Thermal expansion coefficient (1/°C)
f_external = lambda t: 200 * np.sin(0.5 * t)  # External time-varying load (N)

# Constants and parameters for fuel cell model
E_0 = 1.23          # Standard cell potential (V)
R = 8.314           # Universal gas constant (J/(mol·K))
F = 96485           # Faraday constant (C/mol)
T_cell = 298        # Fuel cell temperature (K)
P_H2 = 1            # Partial pressure of hydrogen (atm)
P_O2 = 0.21         # Partial pressure of oxygen (atm)
i_0 = 1e-3          # Exchange current density (A/m²)
R_ohmic = 0.01      # Ohmic resistance (Ω)
alpha_a = 0.5       # Anode transfer coefficient
alpha_c = 0.5       # Cathode transfer coefficient
i_lim = 1           # Limiting current density (A/m²)
A_fuel_cell = 0.01  # Active area of the fuel cell (m²)
V_cell = 0.001      # Volume of the fuel cell (m³)
rho_fuel = 2000     # Density of fuel cell materials (kg/m³)
c_p_fuel = 1000     # Specific heat capacity of fuel cell materials (J/kg·K)

# Initial conditions
initial_conditions = [1e5, 0, 0, 30, 0.5, 0.7, 0]  # [Pressure (Pa), Velocity (m/s), Displacement (m), Temperature (°C), Current Density (A/m²), Control Signal (u), J_prev]

# Differential equations with ESC
def hydraulic_system_with_esc(t, y, V_d, omega, k_s, A_c, F_s, F_c, v_s, m, c, k, K_v, n, C_d, L, D, rho, P_loss, h, A, T_env, c_p, eta_0, T_0, beta, alpha, f_external, E_0, R, F, T_cell, P_H2, P_O2, i_0, R_ohmic, alpha_a, alpha_c, i_lim, A_fuel_cell, V_cell, rho_fuel, c_p_fuel, d, omega_esc, alpha_esc):

    P, v, x, T, i, u_prev, J_prev = y  # Unpack the state variables
    
    # Net Pump Flow Rate
    Delta_P = P  # Pressure difference across the pump
    Q_net = V_d * omega - k_s * Delta_P  # Net flow rate from the pump considering slip
    
    # Non-linear Valve Flow Rate with Dynamic Area
    A_v = K_v * u_prev  # Valve area controlled by u_prev
    Q_v = C_d * A_v * np.sqrt(2 * Delta_P / rho) if Delta_P > 0 else 0  # Flow rate through the valve using orifice equation

    # Cylinder Force
    F_f = F_s * np.sign(v) + (F_c - F_s) * np.exp(-(v / v_s) ** 2)  # Friction force with Stribeck effect
    F_load = m * v + c * v + k * x + f_external(t)  # Dynamic load force including external time-varying load
    F = P * A_c - F_f - F_load  # Net force on the piston
    
    # Cylinder Velocity
    dv_dt = F / m  # Change in velocity (acceleration)
    
    # Pressure Dynamics with Compressibility and Leakage
    dP_dt = (Q_net - Q_v) / (A_c * v if v != 0 else 1e-9)  # Change in pressure, avoid division by zero
    
    # Dynamic Load Force
    dx_dt = v  # Change in displacement (velocity)
    
    # Temperature Effects on Viscosity
    eta = eta_0 * np.exp(-beta * (T - T_0))  # Viscosity as a function of temperature
    
    # Fuel Cell Voltage
    E = E_0 - (R * T_cell / (2 * F)) * np.log(P_H2 / (P_O2 ** 0.5))  # Nernst equation
    i = max(i, 1e-9)
    eta_a = (R * T_cell / (alpha_a * F)) * np.log(i / i_0)  # Activation overpotential
    eta_c = (R * T_cell / (alpha_c * F)) * np.log(i / i_0)  # Activation overpotential
    V = E - eta_a - eta_c - i * R_ohmic  # Cell voltage
    
    # Fuel Cell Power Output
    P_fuel_cell = V * i * A_fuel_cell  # Power output from the fuel cell
    
    # Hydraulic System Power Requirement
    P_required = Q_net * Delta_P + P_loss  # Power required by the hydraulic system
    
    # Update Current Density
    di_dt = (P_required - P_fuel_cell) / (A_fuel_cell * V_cell * rho_fuel * c_p_fuel) if P_fuel_cell != 0 else 0  # Change in current density
    
    # Temperature Dynamics
    dT_dt = (P_loss / (rho * V_d * c_p)) - (h * A * (T - T_env) / (rho * V_d * c_p))  # Change in temperature
    
    # Extremum Seeking Control (ESC) Logic
    # Perturbation signal
        # Perturbation signal
    perturbation = d * np.sin(omega_esc * t)
    
    # Performance measure (objective function), here we aim to minimize power required
    J = P_required
    
    # Derivative of the performance measure
    dJ_dt = (J - J_prev) / (t - (t - 1e-9)) if t > 0 else 0  # Change in performance measure, avoid division by zero
    
    # Adapt control signal using ESC update law
    u = u_prev - alpha_esc * dJ_dt * perturbation
    u = np.clip(u, 0, 1)  # Ensure u stays within the bounds [0, 1]
    
    return [dP_dt, dv_dt, dx_dt, dT_dt, di_dt, u, J]

# Parameters for ESC
d = 0.01  # Amplitude of the perturbation signal
omega_esc = 1.0  # Frequency of the perturbation signal
alpha_esc = 0.01  # Adaptation gain

# Time span for the simulation
t_span = (0, 20)  # Simulation from 0 to 20 seconds
t_eval = np.linspace(*t_span, 2000)  # Evaluate at 2000 points within the time span

# Solve the system of ODEs with ESC
solution = solve_ivp(hydraulic_system_with_esc, t_span, initial_conditions, t_eval=t_eval, 
                     args=(V_d, omega, k_s, A_c, F_s, F_c, v_s, m, c, k, K_v, n, C_d, L, D, rho, P_loss, h, A, T_env, c_p, eta_0, T_0, beta, alpha, f_external, E_0, R, F, T_cell, P_H2, P_O2, i_0, R_ohmic, alpha_a, alpha_c, i_lim, A_fuel_cell, V_cell, rho_fuel, c_p_fuel, d, omega_esc, alpha_esc))

# Extract the results
time = solution.t  # Time points
pressure = solution.y[0]  # Pressure over time
velocity = solution.y[1]  # Velocity over time
displacement = solution.y[2]  # Displacement over time
temperature = solution.y[3]  # Temperature over time
current_density = solution.y[4]  # Current density over time
u_values = solution.y[5]  # Control signal over time
J_values = solution.y[6]  # Performance measure over time

# Plot the results
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(time, pressure)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure vs. Time')

plt.subplot(2, 2, 2)
plt.plot(time, velocity)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs. Time')

plt.subplot(2, 2, 3)
plt.plot(time, displacement)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement vs. Time')

plt.subplot(2, 2, 4)
plt.plot(time, temperature)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs. Time')

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(time, current_density)
plt.xlabel('Time (s)')
plt.ylabel('Current Density (A/m²)')
plt.title('Current Density vs. Time')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(time, u_values)
plt.xlabel('Time (s)')
plt.ylabel('Control Signal (u)')
plt.title('Control Signal vs. Time')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(time, J_values)
plt.xlabel('Time (s)')
plt.ylabel('Performance Measure (J)')
plt.title('Performance Measure vs. Time')
plt.tight_layout()
plt.show()
# Evaluate the final performance
print(f'Final Performance Measure (J): {J_values[-1]}')
