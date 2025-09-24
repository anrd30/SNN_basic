# src/eeg_preprocessing.py
import mne
import numpy as np
import torch
from glob import glob
from typing import List, Tuple, Dict

def load_bci_gdf_files(gdf_paths: List[str],
                       tmin: float = 0.0,
                       tmax: float = 4.0,
                       l_freq: float = 8.0,
                       h_freq: float = 30.0,
                       resample_sfreq: float = None,
                       picks = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load one or more BCI .gdf files, bandpass filter, epoch and return:
      X: [n_epochs, seq_len, n_channels] (torch.float32)
      y: [n_epochs] (torch.long) mapped to 0..C-1
      event_dict: mapping annotation->code returned by mne

    NOTE: adjust tmin/tmax depending on dataset epoch length.
    """
    raws = []
    for p in gdf_paths:
        raw = mne.io.read_raw_gdf(p, preload=True, verbose=False)
        raw.pick(picks) if picks is not None else None
        raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        if resample_sfreq is not None:
            raw.resample(resample_sfreq, npad="auto", verbose=False)
        raws.append(raw)

    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

    # get events from annotations; event_id maps annotation label -> integer
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    # Create epochs (tmin..tmax) — adjust baseline if desired
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                        event_repeated = 'drop',baseline=None, preload=True, verbose=False)

    X = epochs.get_data()   # shape (n_epochs, n_channels, n_times)
    y_codes = epochs.events[:, -1]   # numeric codes

    # Map codes to contiguous 0..C-1 labels (so CrossEntropy works)
    unique_codes = np.unique(y_codes)
    code_to_label = {code: i for i, code in enumerate(sorted(unique_codes))}
    y = np.array([code_to_label[c] for c in y_codes], dtype=np.int64)

    # transpose to [n_epochs, seq_len, n_channels]
    X = np.transpose(X, (0, 2, 1))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    return X_t, y_t, event_dict
