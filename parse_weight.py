import os
import numpy as np

def load_weights(path, tc=False):
    """
    Returns one or two contiguous NumPy arrays (float32).
    If tc=True, also loads 'path + ".tc"' and concatenates.
    """
    # 1) Read the main file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find weights file: {path}")
    with open(path, 'rb') as f:
        # From file => np.fromfile
        base_weights = np.fromfile(f, dtype=np.float32)
        
    return base_weights
