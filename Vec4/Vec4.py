import numpy as np 
import torch

class Vec4:
    def __init__(self,E,px,py,pz):
        self.E = self._ensure_array(E)

    def _ensure_array(self,x):
        if isinstance(x, (int,float)):
            return np.array(x)
        elif isinstance(x, (np.ndarray, torch.Tensor)):
            return x
        elif isinstance(x, (list,tuple)):
            return mp.array(x)
        else:
            raise TypeError("Input must be either a scalar, list, nupy array, or torch tensor")

    def _detect_backend(self,x):
        if isinstance(x, torch.Tensor):
            return "torch"
        elif isinstance(x, np.ndarray):
            return "numpy"
        else:
            raise TypeError("Unknown backend")

    def _op(self, np_func, torch_func, *args):
        if self.backend == "numpy":
            return np_func(*args)
        elif self.backend == "torch":
            return torch_func(*args)
        else:
            raise TypeError("No function found for backend")


