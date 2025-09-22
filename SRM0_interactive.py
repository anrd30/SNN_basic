import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- 1. Define the Kernel Functions ---

def eta_kernel(s, threshold, tau_m):
    """Refractory effect (voltage drop after spike)."""
    if s < 0 or tau_m == 0: return 0
    return -threshold * np.exp(-s / tau_m)

def epsilon_kernel(s, tau_s):
    """Postsynaptic potential (EPSP) kernel."""
    if s < 0 or tau_s == 0: return 0
    return (s / tau_s) * np.exp(-s / tau_s)

# --- 2. Main Simulation Function ---

def run_simulation(N, threshold, tau_m, tau_s, total_time, dt, W_strength, input_rate, beta):
    """Runs a full simulation of the SRM population."""
    time_steps = np.arange(0, total_time, dt)
    
    # Generate Random Input Spikes for Each Neuron
    prob_per_step = input_rate * (dt / 1000.0)
    input_spikes = []
    for n in range(N):
        neuron_spikes = time_steps[np.random.rand(len(time_steps)) < prob_per_step]
        input_spikes.append(neuron_spikes)

    # Synaptic weight matrix
    W = W_strength * np.ones((N, N))
    np.fill_diagonal(W, 0)
    
    # Initialize State Variables
    last_spike_time = np.full(N, -1000.0)
    output_spikes = [[] for _ in range(N)]

    # Simulation Loop
    for i, t in enumerate(time_steps):
        for n in range(N):
            # Refractory effect
            refractory = eta_kernel(t - last_spike_time[n], threshold, tau_m)
            
            # Input effect from this neuron's unique spike train
            input_effect = sum(epsilon_kernel(t - s, tau_s) for s in input_spikes[n] if t > s)
            
            # Recurrent input from other neurons in the network
            recurrent_input = 0
            for m in range(N):
                if m != n:
                    recurrent_input += sum(
                        W[n, m] * epsilon_kernel(t - spike_t, tau_s)
                        for spike_t in output_spikes[m] if spike_t < t
                    )
            
            # Total membrane potential
            u = refractory + input_effect + recurrent_input
            
            # Stochastic escape firing
            firing_prob = dt * np.exp(beta * (u - threshold))
            if np.random.rand() < firing_prob:
                output_spikes[n].append(t)
                last_spike_time[n] = t
    
    return output_spikes, input_spikes


# --- 3. Setup the Interactive Plot ---
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial parameters
N = 10
threshold = 1.0
total_time = 200
dt = 0.1
beta = 1.5

# Create axes for the sliders
ax_tau_m = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_tau_s = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_w = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_rate = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_reset = plt.axes([0.05, 0.05, 0.15, 0.04])

# Create slider objects
slider_tau_m = Slider(ax_tau_m, 'τm (Membrane τ)', 5.0, 40.0, valinit=20.0)
slider_tau_s = Slider(ax_tau_s, 'τs (Synaptic τ)', 1.0, 15.0, valinit=5.0)
slider_w = Slider(ax_w, 'W Strength', -0.1, 0.2, valinit=0.05)
slider_rate = Slider(ax_rate, 'Input Rate (Hz)', 0, 50, valinit=10)
button_reset = Button(ax_reset, 'Reset Inputs')

# Global variable to hold input spikes so they don't change on every slider move
current_input_spikes = None

# --- 4. The Update Function ---
def update(val=None):
    global current_input_spikes
    
    # Rerun simulation with current slider values
    output_spikes, new_input_spikes = run_simulation(
        N=N, threshold=threshold,
        tau_m=slider_tau_m.val,
        tau_s=slider_tau_s.val,
        total_time=total_time, dt=dt,
        W_strength=slider_w.val,
        input_rate=slider_rate.val,
        beta=beta
    )
    
    # If this is the first run, or reset was clicked, use the new spikes
    if current_input_spikes is None:
        current_input_spikes = new_input_spikes

    # Clear the previous plot
    ax.cla()
    
    # Plot the new raster
    for n in range(N):
        if len(current_input_spikes[n]) > 0:
            ax.plot(current_input_spikes[n], np.full_like(current_input_spikes[n], n + 1), 'g.', markersize=2, label='Input Spikes' if n==0 else "")
        if len(output_spikes[n]) > 0:
            ax.vlines(output_spikes[n], n + 0.5, n + 1.5, 'b', label='Output Spikes' if n==0 else "")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title('Interactive Population Raster Plot')
    ax.set_ylim(0.5, N + 1.5)
    ax.legend(loc='upper right')
    fig.canvas.draw_idle()

def reset_inputs(event):
    global current_input_spikes
    current_input_spikes = None # Signal to generate new spikes on next update
    update()

# --- 5. Connect Widgets and Run ---
slider_tau_m.on_changed(update)
slider_tau_s.on_changed(update)
slider_w.on_changed(update)
slider_rate.on_changed(update)
button_reset.on_clicked(reset_inputs)

# Initial plot
reset_inputs(None)
plt.show()
