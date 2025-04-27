import h5py
import numpy as np

filepath = r'G:\My Drive\FinalProject\Data\Patient_01\LFP_signals\CSC1_LFP.mat'

with h5py.File(filepath, 'r') as f:
    refs = f['#refs#']
    data_ref = refs['f']['data'][0][0]   # שולף את ההפניה

    # מבצע dereference
    actual_data = f[data_ref]

    print(f"[DEBUG] Dereferenced data shape: {actual_data.shape}")
    print(f"[DEBUG] Dereferenced data dtype: {actual_data.dtype}")

    # נטען לדוגמה 10 ערכים ראשונים
    data_array = np.array(actual_data)
    print(f"[DEBUG] Data sample: {data_array.flatten()[:10]}")