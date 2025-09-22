import snntorch as snn
import torch
import matplotlib.pyplot as plt

# Parameters
time_step = 1e-3  # 1 ms
R = 5
C = 1e-3
num_steps = 100

# Initialize LIF neuron (Lapicque model)
lif1 = snn.Lapicque(R=R, C=C, time_step=time_step, threshold=0.4, reset_mechanism="zero")

# Initial states
mem = torch.zeros(1)
spk_out = torch.zeros(1)

# Input current: off for 10 ms, then constant 0.2
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(90, 1) * 0.2), 0)

# Recording
mem_rec = [mem]
spk_out_rec = [spk_out]

# Simulation loop
for i in range(num_steps):
    spk_out, mem = lif1(cur_in[i], mem)
    mem_rec.append(mem)
    spk_out_rec.append(spk_out)

# Convert to tensors
mem_rec = torch.stack(mem_rec)
spk_out_rec = torch.stack(spk_out_rec)

# Time axis in ms
time_axis = torch.arange(0, num_steps + 1) * time_step * 1000

# Plot
plt.figure(figsize=(12, 8))
plt.plot(time_axis, mem_rec.detach().numpy(), label='Membrane Potential')
plt.scatter(time_axis, spk_out_rec.detach().numpy(), color='orange', label='Spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential / Spike')
plt.legend()
plt.show()
