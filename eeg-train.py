
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from rnn.rnn_baseline import RNNBaseline
from eeg_preprocessing import load_bci_gdf_files
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
# Hyperparameters
SEQ_TMIN = 0.0
SEQ_TMAX = 4.0
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_GLOB = "data/*.gdf"
MODEL_OUT = "rnn_eeg.pth"

def main():
    print("Device:", DEVICE)

    gdf_paths = sorted(list(__import__("glob").glob(DATA_GLOB)))
    if len(gdf_paths) == 0:
        raise RuntimeError(f"No .gdf files found with pattern {DATA_GLOB}. Put files in data/")

    print("Loading and preprocessing EEG files (this can take a few minutes)...")
    X, y, event_dict = load_bci_gdf_files(
        gdf_paths, tmin=SEQ_TMIN, tmax=SEQ_TMAX, l_freq=8., h_freq=30., resample_sfreq=128
    )
    X = (X - X.mean(dim=(0,1), keepdim=True)) / (X.std(dim=(0,1), keepdim=True) + 1e-6)

    print("Epochs loaded:", X.shape, "labels:", torch.unique(y))
    print("Class distribution:", torch.bincount(y))


    X_flat = X.reshape(X.shape[0], -1).numpy()
    y_np = y.numpy()

    # Apply oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y_np)

# Reshape X back to original shape
    X_resampled = torch.tensor(X_resampled).view(-1, X.shape[1], X.shape[2])
    y_resampled = torch.tensor(y_resampled)

    # Weighted loss
    '''class_counts = torch.bincount(y)
    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))'''
    criterion = nn.CrossEntropyLoss()

    n_channels = X.shape[2]
    n_classes = int(torch.unique(y).numel())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    model = RNNBaseline(input_dim=n_channels, hidden_dim=HIDDEN_DIM, output_dim=n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total

        # Evaluation on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                preds = logits.argmax(dim=1)
                test_correct += (preds == yb).sum().item()
                test_total += yb.size(0)
        test_acc = test_correct / test_total

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} - train acc: {epoch_acc:.3f} - test acc: {test_acc:.3f}")
        scheduler.step()

    torch.save(model.state_dict(), MODEL_OUT)
    print("Saved EEG RNN model to", MODEL_OUT)

if __name__ == "__main__":
    main()


