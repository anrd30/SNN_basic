import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from rnn.utils import generate_sine_data

# ===============================
# RNN Model Definition
# ===============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from rnn.utils import generate_sine_data

# ===============================
# RNN Model Definition
# ===============================
class RNNBaseline(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(RNNBaseline, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim,num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        self.dropout = nn.Dropout(p=0.5)
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1) 
        pooled = self.dropout(pooled)
        out = self.fc(pooled)         # [batch, seq_len, output_dim]
        return out

# ===============================
# Training Function
# ===============================
def train_rnn():
    # Hyperparameters
    seq_len = 20
    num_sequences = 1000
    batch_size = 32
    hidden_dim = 32
    epochs = 30
    lr = 0.001

    # 1. Generate data
    print("Generating sine wave data...")
    X, Y = generate_sine_data(seq_len=seq_len, num_sequences=num_sequences)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNBaseline(input_dim=1, hidden_dim=hidden_dim, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # 4. Save the trained model
    torch.save(model.state_dict(), "rnn_baseline.pth")
    print("Model saved to rnn_baseline.pth")

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    train_rnn()


# ===============================
# Training Function
# ===============================
def train_rnn():
    # Hyperparameters
    seq_len = 20
    num_sequences = 1000
    batch_size = 32
    hidden_dim = 32
    epochs = 30
    lr = 0.001

    # 1. Generate data
    print("Generating sine wave data...")
    X, Y = generate_sine_data(seq_len=seq_len, num_sequences=num_sequences)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNBaseline(input_dim=1, hidden_dim=hidden_dim, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # 4. Save the trained model
    torch.save(model.state_dict(), "rnn_baseline.pth")
    print("Model saved to rnn_baseline.pth")

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    train_rnn()
