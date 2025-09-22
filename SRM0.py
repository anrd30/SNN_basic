import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Kernel Functions ---

def eta_kernel(s, threshold, tau_m):
    """Refractory effect (voltage drop after spike)."""
    return 0 if s < 0 else -threshold * np.exp(-s / tau_m)

def epsilon_kernel(s, tau_s):
    """Postsynaptic potential (EPSP) kernel."""
    return 0 if s < 0 else (s / tau_s) * np.exp(-s / tau_s)

# --- 2. Simulation Parameters ---

N = 10                 # number of neurons
threshold = 1.0         # firing threshold
tau_m = 20.0            # refractory time constant (ms)
tau_s = 5.0             # synaptic time constant (ms)
total_time = 200        # total simulation time (ms)
dt = 0.1                # time step (ms)
time_steps = np.arange(0, total_time, dt)

# Stochastic escape noise parameter
beta = 1.5  # higher = more deterministic

# Synaptic weight matrix (all-to-all, zero diagonal)
W = 0.05 * np.ones((N, N))
np.fill_diagonal(W, 0)

# Input spikes (can be neuron-specific)
input_spikes = [20, 80, 85, 120, 160, 162, 165]

# --- 3. Initialize State Variables ---

voltage = np.zeros((N, len(time_steps)))
last_spike_time = np.full(N, -1000.0)
output_spikes = [[] for _ in range(N)]

# --- 4. Simulation Loop ---

print("Running population simulation...")
for i, t in enumerate(time_steps):
    for n in range(N):
        # Refractory effect
        time_since_last = t - last_spike_time[n]
        refractory = eta_kernel(time_since_last, threshold, tau_m)
        
        # Input effect
        input_effect = sum(epsilon_kernel(t - s, tau_s) for s in input_spikes if t > s)
        
        # Recurrent input
        recurrent_input = 0
        for m in range(N):
            if m != n:
                recurrent_input += sum(
                    W[n, m] * epsilon_kernel(t - spike_t, tau_s)
                    for spike_t in output_spikes[m] if spike_t < t
                )
        
        # Total membrane potential
        u = refractory + input_effect + recurrent_input
        voltage[n, i] = u
        
        # Stochastic escape firing
        firing_prob = dt * np.exp(beta * (u - threshold))
        if np.random.rand() < firing_prob:
            output_spikes[n].append(t)
            last_spike_time[n] = t
            voltage[n, i] = 2.0  # spike peak for visualization

print("Simulation complete!")

# --- 5. Plot Raster Plot ---

plt.figure(figsize=(12, 6))
for n in range(N):
    plt.vlines(output_spikes[n], n + 0.5, n + 1.5)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Population Raster Plot')
plt.show()

# --- 6. Plot Population Firing Rate ---

firing_rate = np.zeros(len(time_steps))
for n in range(N):
    for spike_t in output_spikes[n]:
        idx = int(spike_t / dt)
        firing_rate[idx] += 1
firing_rate /= N  # normalize by population size

plt.figure(figsize=(12, 4))
plt.plot(time_steps, firing_rate)
plt.xlabel('Time (ms)')
plt.ylabel('Population Firing Rate')
plt.title('Population Activity')
plt.show()
