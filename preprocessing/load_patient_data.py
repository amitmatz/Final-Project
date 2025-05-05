import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder


def load_patient_data(patient_name, fixed_length=1000):
    """
    Load processed data for a patient and return a PyTorch-compatible dataset.

    Parameters:
        patient_name (str): Name of the patient (used to find the .npy file)
        fixed_length (int): Number of samples per signal (padding/cropping)

    Returns:
        dataset (TensorDataset): PyTorch dataset (X, y)
        label_encoder (LabelEncoder): To decode predictions
        electrode_map (dict): Electrode name to channel index
    """
    # Load saved data
    data_path = os.path.join(PROCESSED_DATA_DIR, f"{patient_name}_data.npy")
    data_list = np.load(data_path, allow_pickle=True)

    signals, labels, electrodes = [], [], []
    for item in data_list:
        signal = item['signal']
        label = item['label']
        electrode = item['electrode']

        # Pad or crop to fixed length
        if len(signal) < fixed_length:
            padded = np.pad(signal, (0, fixed_length - len(signal)), mode='constant')
        else:
            padded = signal[:fixed_length]

        signals.append(padded)
        labels.append(label)
        electrodes.append(electrode)

    # Encode labels (a, e, i, o, u → 0-4)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Map electrodes to channel numbers
    unique_electrodes = sorted(set(electrodes))
    electrode_map = {name: i for i, name in enumerate(unique_electrodes)}
    channels = np.array([electrode_map[e] for e in electrodes])

    # Convert to torch tensors
    X = torch.tensor(signals, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.long)
    ch = torch.tensor(channels, dtype=torch.long)

    # Combine inputs: [signal + channel info]
    # Optionally, you can use channel as an extra input if needed

    dataset = TensorDataset(X, y, ch)
    return dataset, label_encoder, electrode_map
