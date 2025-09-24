import torch
from torch.utils.data import TensorDataset, DataLoader
from rnn.rnn_baseline import RNNBaseline
from eeg_preprocessing import load_bci_gdf_files
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Hyperparameters
SEQ_TMIN = 0.0
SEQ_TMAX = 4.0
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "rnn_eeg.pth"
DATA_GLOB = "data/*.gdf"

def main():
    print("Device:", DEVICE)

    gdf_paths = sorted(list(__import__("glob").glob(DATA_GLOB)))
    if len(gdf_paths) == 0:
        raise RuntimeError(f"No .gdf files found with pattern {DATA_GLOB}")

    print("Loading EEG data...")
    X, y, _ = load_bci_gdf_files(
        gdf_paths, tmin=SEQ_TMIN, tmax=SEQ_TMAX, l_freq=8., h_freq=30., resample_sfreq=128
    )
    X = (X - X.mean(dim=(0,1), keepdim=True)) / (X.std(dim=(0,1), keepdim=True) + 1e-6)

    # Split test set (same as training script)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Load model
    n_channels = X.shape[2]
    n_classes = int(torch.unique(y).numel())
    model = RNNBaseline(input_dim=n_channels, hidden_dim=256, output_dim=n_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Collect predictions
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            preds = logits.argmax(dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n✅ Balanced Accuracy: {bal_acc:.3f}")
    print(f"✅ Macro F1-score: {macro_f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()