import torch
import torch.nn as nn
import numpy as np
from utils import generate_sine_data
from rnn_baseline import RNNBaseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNN_WEIGHTS_PATH = "rnn_baseline.pth"
SEQ_LEN = 20
BATCH_SIZE = 64
NUM_SAMPLES = 512

#SNN paramters
dt  = 1.0
tau = 5.0
alpha = dt/tau
V_th = 1.0
V_reset = 0.2
scale_input = 10.0 

torch.manual_seed(0)
np.random.seed(0)#ensures randomly generated data is the same across runs

#helper : run the RNN model

def run_rnn(model, X):
    # X : tensor of shape (batch_size, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        out = model(X.to(DEVICE)).cpu()
    return out

#helper : run the SNN model

def run_snn(W_in, U_rec, fc_weight, fc_bias, X, dt=1.0, tau=5.0, V_th=1.0, V_reset=0.0, alpha=None):
    """
    Vectorized SNN forward pass using PyTorch.
    Preserves LIF dynamics with hard or soft reset.
    """
    if alpha is None:
        alpha = dt / tau

    N, seq_len, input_dim = X.shape
    hidden_dim = W_in.shape[0]
    output_dim = fc_weight.shape[0]

    # Move to device
    device = X.device
    W_in, U_rec, fc_weight, fc_bias = W_in.to(device), U_rec.to(device), fc_weight.to(device), fc_bias.to(device)

    # Preallocate tensors
    V = torch.zeros(N, hidden_dim, device=device)
    s = torch.zeros(N, hidden_dim, device=device)
    spike_record = torch.zeros(N, seq_len, hidden_dim, device=device)

    # Precompute input current for all timesteps
    I_in_seq = torch.matmul(X * scale_input, W_in.T)  # shape: (N, seq_len, hidden_dim)

    # Loop over time steps (small loop, only for recurrent contribution)
    for t in range(seq_len):
        I_in = I_in_seq[:, t, :]
        I_rec = torch.matmul(s, U_rec.T)
        I = I_in + I_rec

        # Update membrane potential
        V = (1 - alpha) * V + alpha * I

        # Generate spikes
        s = (V >= V_th).float()
        spike_record[:, t, :] = s

        # Apply reset
        if V_reset == 0.0:
            V = V * (1 - s)  # hard reset
        else:
            V = V * (1 - s) - s * V_reset  # soft reset

    # Compute firing rates across time
    rates = spike_record.mean(dim=1)  # (N, hidden_dim)

    # Decode output
    decoded = torch.matmul(rates, fc_weight.T) + fc_bias

    return decoded.cpu(), spike_record.cpu()

def measure_compute(model, W_in, U_rec, fc_weight, fc_bias, X_batch, dt=1.0, tau=5.0, V_th=1.0, V_reset=0.0, alpha=None):
    import torch.cuda as cuda

    # --- RNN Timing ---
    cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    rnn_out = run_rnn(model, X_batch)
    end.record()
    torch.cuda.synchronize()
    rnn_time = start.elapsed_time(end)  # milliseconds
    rnn_mem = cuda.max_memory_allocated() / 1e6

    # --- SNN Timing ---
    cuda.reset_peak_memory_stats()
    start.record()
    snn_decoded, spike_record = run_snn(W_in, U_rec, fc_weight, fc_bias, X_batch, dt, tau, V_th, V_reset, alpha)
    end.record()
    torch.cuda.synchronize()
    snn_time = start.elapsed_time(end)
    snn_mem = cuda.max_memory_allocated() / 1e6

    # --- Estimate FLOPs ---
    batch_size, seq_len, input_dim = X_batch.shape
    hidden_dim = W_in.shape[0]
    output_dim = fc_weight.shape[0]

    # RNN FLOPs per batch (rough approx)
    rnn_flops = 2 * batch_size * seq_len * (hidden_dim*input_dim + hidden_dim*hidden_dim)

    # SNN FLOPs: scaled by avg spike rate
    avg_spike_rate = spike_record.mean().item()
    snn_flops = 2 * batch_size * seq_len * hidden_dim * input_dim * avg_spike_rate \
                + 2 * batch_size * seq_len * hidden_dim * hidden_dim * avg_spike_rate

    return rnn_time, snn_time, rnn_mem, snn_mem, rnn_flops, snn_flops, avg_spike_rate





def main():
    print("Device: ",DEVICE)


    #1)cREATE rnn MODEL AND LOAD WEIGHTS
    model = RNNBaseline(input_dim=1,hidden_dim=32,output_dim=1).to(DEVICE)
    state = torch.load(RNN_WEIGHTS_PATH,map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("RNN weights loaded from: ",RNN_WEIGHTS_PATH)

    #2)EXTRACT WEIGHTS
    W_in = model.rnn.weight_ih_l0.detach().clone().cpu()
    U_rec = model.rnn.weight_hh_l0.detach().clone().cpu()

    if hasattr(model.rnn, "bias_ih_l0") and hasattr(model.rnn, "bias_hh_l0"):
        b_ih = model.rnn.bias_ih_l0.detach().clone().cpu()
        b_hh = model.rnn.bias_hh_l0.detach().clone().cpu()
        rnn_bias = (b_ih + b_hh).unsqueeze(0)  # shape [1, hidden_dim]
    else:
        rnn_bias = torch.zeros(1, W_in.shape[0])


    fc_weight = model.fc.weight.detach().clone().cpu()
    fc_bias = model.fc.bias.detach().clone().cpu()

    X_all,Y_all = generate_sine_data(seq_len=SEQ_LEN,num_sequences=NUM_SAMPLES)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_all).float(),torch.tensor(Y_all).float())
    loader  = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False)

    mse_rnn = nn.MSELoss()
    total_mse_rnn = 0.0
    total_mse_snn = 0.0
    n_batches = 0


    for X_batch,Y_batch in loader:
        n_batches += 1
        #RNN
        rnn_out = run_rnn(model,X_batch)
        rnn_decoded = rnn_out[:,-1,:]  # get the last time step output
        target = Y_batch[:,-1,:]

        #compute RNN MSE
        mse_r = mse_rnn(rnn_decoded,target).item()
        total_mse_rnn += mse_r

        #Run SNN using RNN weights
        snn_decoded,spike_record = run_snn(W_in,U_rec,fc_weight,fc_bias,X_batch,dt=dt,tau=tau,V_th=V_th,V_reset=V_reset,alpha=alpha)
        
        snn_target =target
        mse_s = mse_rnn(snn_decoded,snn_target).item()
        total_mse_snn += mse_s
        rnn_time, snn_time, rnn_mem, snn_mem, rnn_flops, snn_flops, avg_spike_rate = measure_compute(
        model, W_in, U_rec, fc_weight, fc_bias, X_batch, dt=dt, tau=tau, V_th=V_th, V_reset=V_reset, alpha=alpha
        )

        print(f"Batch {n_batches}: RNN time={rnn_time:.2f}ms, SNN time={snn_time:.2f}ms, Avg spike rate={avg_spike_rate:.2f}")
        print(f"Memory: RNN={rnn_mem:.1f}MB, SNN={snn_mem:.1f}MB")
        print(f"FLOPs (approx): RNN={rnn_flops:.1e}, SNN={snn_flops:.1e}")


    

    avg_mse_rnn = total_mse_rnn/n_batches
    avg_mse_snn = total_mse_snn/n_batches

        
    print(f"Average RNN MSE over {n_batches} batches: {avg_mse_rnn:.6f}")
    print(f"Average SNN MSE over {n_batches} batches: {avg_mse_rnn:.6f}")


    X_vis, Y_vis = next(iter(loader))
    _, spike_vis = run_snn(W_in,U_rec,fc_weight,fc_bias,X_vis,dt=dt,tau=tau,V_th=V_th,V_reset=V_reset,alpha=alpha)

    import matplotlib.pyplot as plt
    sp =spike_vis.numpy()
    plt.figure(figsize=(10,4))
    example = 0
    plt.imshow(sp[example].T,aspect='auto',cmap='gray_r')
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title("Spiking activity of SNN neurons")
    plt.colorbar(label='Spike (1) or No Spike (0)')
    plt.show()



if __name__ == "__main__":
        print("Starting SNN conversion script...")
        main()

