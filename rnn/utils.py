# src/utils.py
import numpy as np
import torch

def generate_sine_data(seq_len=200, num_sequences=100, freq=0.03):
    """
    Generate batches of sine wave sequences for RNN training.
    """
    X = []
    Y = []
    for _ in range(num_sequences):
        start = np.random.randint(0, 50)
        t = np.arange(start, start + seq_len + 1)
        sine_wave = np.sin(2 * np.pi * freq * t)
        
        # Inputs: sine wave[:-1], Targets: sine wave[1:]
        X.append(sine_wave[:-1])
        Y.append(sine_wave[1:])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, input_dim=1]
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, output_dim=1]
    return X, Y
