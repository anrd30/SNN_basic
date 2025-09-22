import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Spiking Neuron and Network Parameters ---
N = 1000                # Number of neurons in the network
dt = 0.1                # Simulation time step (ms)
sim_time = 1000           # Total simulation time (ms)
train_time = 800        # Time to train the network (ms)

# LIF Neuron Parameters
tau_mem = 10.0          # Membrane time constant (ms)
v_reset = -65.0         # Reset potential (mV)
v_threshold = -50.0     # Firing threshold (mV)

# Synaptic Parameters (Double Exponential Filter)
tau_rise = 2.0          # Synaptic rise time constant (ms)
tau_decay = 20.0        # Synaptic decay time constant (ms)

# --- 2. Define the Target Signal (Supervisor) ---
freq = 5                # Frequency of the target sine wave in Hz
target_signal = np.sin(2 * np.pi * freq * np.arange(0, sim_time, dt) / 1000.0)

# --- 3. Set Up the Network and FORCE Learning Parameters ---

# FORCE training parameters
G = 1.0                 # Controls the strength of the chaotic recurrent matrix
Q = 1.5                 # Controls the strength of the learning feedback
alpha = 1.0             # Learning rate parameter for RLS

# Initialize the static, chaos-inducing weight matrix (omega_0)
p_connect = 0.1         # Sparsity of the connection matrix
omega_0 = np.random.randn(N, N) * G / (N * p_connect)**0.5
# Create a mask for sparse connections
connectivity_mask = np.random.rand(N, N) < p_connect
omega_0[~connectivity_mask] = 0

# Initialize the learned components
# phi: decoders (learned)
phi = np.zeros(N)
# eta: encoders (static random feedback weights)
eta = np.random.uniform(-1, 1, N)
# P: inverse correlation matrix for RLS
P = np.eye(N) / alpha

# Initialize neuron state variables
v = np.full(N, v_reset) # Membrane potentials
r = np.zeros(N)         # Filtered spike train (synaptic current)
h = np.zeros(N)         # Helper variable for synaptic filter

# Storage for plotting
network_output = np.zeros(int(sim_time / dt))
spike_raster = []

# --- 4. The Main Simulation Loop ---

print("Starting simulation...")
time_steps = int(sim_time / dt)
train_steps = int(train_time / dt)

for i in range(time_steps):
    t = i * dt
    
    # --- A. Calculate Network Output ---
    x_hat = np.dot(phi, r)
    network_output[i] = x_hat
    
    # --- B. RLS Update (Training Phase) ---
    if i < train_steps:
        # Calculate error
        error = x_hat - target_signal[i]
        
        # Update P matrix
        Pr = np.dot(P, r)
        P -= np.outer(Pr, Pr) / (1 + np.dot(r, Pr))
        
        # Update decoders (phi)
        phi -= error * np.dot(P, r)

    # --- C. Simulate Network Dynamics ---
    
    # Calculate total synaptic input current for this time step
    # This includes the chaotic matrix and the learned feedback loop
    total_input = np.dot(omega_0, r) + Q * eta * x_hat

    # Update synaptic filter variables (r and h)
    h += dt * (-h + r) / tau_rise
    r += dt * (-r + h) / tau_decay

    # Update membrane potential
    dv = (-(v - v_reset) + total_input) / tau_mem
    v += dt * dv
    
    # Check for spikes
    spiking_neurons = v >= v_threshold
    
    # Store spikes for raster plot
    if np.any(spiking_neurons):
        spike_indices = np.where(spiking_neurons)[0]
        for neuron_idx in spike_indices:
            spike_raster.append([t, neuron_idx])
            
    # Reset voltage of spiking neurons and add spike to synaptic filter
    v[spiking_neurons] = v_reset
    # Add a delta pulse for each spike to the synaptic current
    r[spiking_neurons] += 1.0 / (tau_rise * tau_decay)
    
    if i % 1000 == 0:
        print(f"  Time: {t:.0f} ms / {sim_time} ms")

print("Simulation complete.")

# --- 5. Plot the Results ---

# Convert spike raster to numpy array for plotting
spike_raster = np.array(spike_raster)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Network Output vs. Target Signal
ax1.set_title("FORCE-Trained Spiking Network", fontsize=16)
ax1.plot(np.arange(0, sim_time, dt), target_signal, 'k--', label='Target Signal', lw=2)
ax1.plot(np.arange(0, sim_time, dt), network_output, 'b', label='Network Output', lw=1.5)
ax1.axvline(train_time, color='r', linestyle=':', label='Training Ends')
ax1.set_ylabel("Output", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot 2: Spike Raster
ax2.set_title("Spiking Activity", fontsize=14)
if len(spike_raster) > 0:
    ax2.scatter(spike_raster[:, 0], spike_raster[:, 1], s=2, c='k', marker='|')
ax2.set_ylabel("Neuron Index", fontsize=12)
ax2.set_xlabel("Time (ms)", fontsize=12)
ax2.set_ylim(0, N)
ax2.set_xlim(0, sim_time)

plt.tight_layout()
plt.show()
