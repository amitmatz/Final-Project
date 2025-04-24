import h5py
import numpy as np

def print_h5_structure(filepath):
    with h5py.File(filepath, 'r') as f:
        print("📁 File structure:")
        f.visititems(lambda name, obj: print(f"🔹 {name} ({type(obj).__name__})"))

def read_all_variables(filepath):
    with h5py.File(filepath, 'r') as f:
        print("\n📦 Reading content:")
        for key in f.keys():
            try:
                data = np.array(f[key])
                print(f"\n🔹 {key}:")
                if data.dtype.kind == 'S':
                    # מחרוזות
                    decoded = [x.decode('utf-8') for x in data.flatten()]
                    print(decoded)
                else:
                    print(data)
            except Exception as e:
                print(f"⚠️ Could not read {key}: {e}")

if __name__ == "__main__":
    path = "sound_w_times.mat"
    print_h5_structure(path)
    read_all_variables(path)
