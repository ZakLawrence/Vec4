import numpy as np
import torch

def ensure_array(x):
    if isinstance(x, (int,float)):
        return np.array(x)
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x
    elif isinstance(x, (list,tuple)):
        return np.array(x)
    else:
        raise TypeError("Input must be either a scalar, list, nupy array, or torch tensor")

def detect_backend(x):
    if isinstance(x, (torch.Tensor,torch.dtype)) or torch.is_tensor(x):
        return "torch"
    elif isinstance(x, (np.ndarray,np.generic)):
        return "numpy"
    else:
        raise TypeError("Unknown backend")

def op(np_func, torch_func, *args):
    backends = {detect_backend(arg) for arg in args}
    if len(backends) > 1:
        raise ValueError(f"Mixed backends in args: {backends}")
    backend = backends.pop()
    if backend == "numpy":
        return np_func(*args)
    elif backend == "torch":
        return torch_func(*args)
    else:
        raise TypeError("No function found for backend")