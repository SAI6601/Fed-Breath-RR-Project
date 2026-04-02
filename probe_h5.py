"""Quick probe of HDF5 CapnoBase .mat file structure."""
import h5py
import numpy as np

f = h5py.File(r'data\capnobase\0009_8min.mat', 'r')

print("=== FULL STRUCTURE ===")
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET: {name}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  GROUP:   {name}/")
f.visititems(show)

# Try to read PPG signal
print("\n=== PPG SIGNAL ===")
for path in ['signal/pleth/y', 'signal/pleth/v', 'signal/ppg/y', 'signal/ppg']:
    if path in f:
        obj = f[path]
        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj).ravel()
            print(f"  {path}: shape={obj.shape}, len={len(arr)}, range=[{arr.min():.3f}, {arr.max():.3f}]")
        else:
            print(f"  {path}: type={type(obj).__name__}, keys={list(obj.keys()) if isinstance(obj, h5py.Group) else 'N/A'}")

# Try to read RR labels
print("\n=== RR LABELS ===")
for path in ['reference/rr/co2/x', 'reference/rr/co2/y',
             'reference/rr/x', 'reference/rr/y',
             'labels/rr', 'labels/rr/co2']:
    if path in f:
        obj = f[path]
        if isinstance(obj, h5py.Dataset):
            try:
                arr = np.array(obj).ravel()
                arr_finite = arr[np.isfinite(arr)]
                print(f"  {path}: shape={obj.shape}, len={len(arr)}, mean={np.mean(arr_finite):.2f}, range=[{arr_finite.min():.2f}, {arr_finite.max():.2f}]")
            except Exception as e:
                print(f"  {path}: Dataset but error reading: {e}")
        else:
            print(f"  {path}: type={type(obj).__name__}")

# Try param/samplingrate
print("\n=== PARAM ===")
if 'param' in f:
    obj = f['param']
    if isinstance(obj, h5py.Group):
        print(f"  param keys: {list(obj.keys())}")
        for k in obj.keys():
            sub = obj[k]
            if isinstance(sub, h5py.Dataset):
                try:
                    val = np.array(sub).ravel()
                    print(f"    param/{k} = {val}")
                except:
                    print(f"    param/{k}: shape={sub.shape}, dtype={sub.dtype}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  param: dataset shape={obj.shape}")

# Top-level keys
print(f"\n=== TOP-LEVEL KEYS: {list(f.keys())} ===")

f.close()
print("\nDone.")
