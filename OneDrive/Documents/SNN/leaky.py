import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

num_input  =784
num_hidden = 1000
num_output = 10
beta = 0.99

fc1 = torch.nn.Linear(num_input, num_hidden)
fc2 = torch.nn.Linear(num_hidden, num_output)

lif1 = snn.Leaky(beta=beta)
lif2 = snn.Leaky(beta=beta)

mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
print(f"Dimensions of spk_in: {spk_in.size()}")

for i in range(100):
    curr1 = fc1(spk_in[i])
    spk1, mem1 = lif1(curr1, mem1)
    spk2, mem2 = lif2(fc2(spk1), mem2)
    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

plt.figure(figsize=(12, 8))
splt.rasterplot(spk1_rec)
splt.rasterplot(spk2_rec)
plt.show()