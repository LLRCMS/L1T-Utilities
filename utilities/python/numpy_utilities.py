import numpy as np

def find_closest(array, target):
    # array must be sorted
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array)-1)
    left = a[idx-1]
    right = a[idx]
    idx -= target - left < right - target
    return idx
