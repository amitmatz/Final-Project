import h5py

with h5py.File("CSC4_LFP.mat", "r") as f:
    def print_structure(name, obj):
        print(name)

    print("🔍 File structure:")
    f.visititems(print_structure)
