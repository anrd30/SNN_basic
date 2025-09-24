''''import mne

# Make sure the path matches your folder structure
raw = mne.io.read_raw_gdf("data/A01T.gdf", preload=True, verbose=True)

print(raw.info)
print("Channels:", raw.info["ch_names"])
print("Sampling frequency:", raw.info["sfreq"])'''

import torch
print("Cuda available:", torch.cuda.is_available())
print("get device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")