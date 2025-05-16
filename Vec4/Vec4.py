import numpy as np 
import torch

class Vec4:
    def __init__(self,E,px,py,pz):
        self.E = self._ensure_array(E)
        self.px = self._ensure_array(px)
        self.py = self._ensure_array(py)
        self.pz = self._ensure_array(pz)

        if not (self.E.shape == self.px.shape == self.py.shape == self.pz.shape):
            raise ValueError("All components must have the same shape")
        self.backend = self._detect_backend(self.E)

    def _ensure_array(self,x):
        if isinstance(x, (int,float)):
            return np.array(x)
        elif isinstance(x, (np.ndarray, torch.Tensor)):
            return x
        elif isinstance(x, (list,tuple)):
            return np.array(x)
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
        
    @property
    def mass(self):
        E2 = self.E**2
        p2 = self.px**2 + self.py**2 + self.pz**2

        diff = E2 - p2
        return self._op(
            np.sqrt, torch.sqrt,
            np.maximum(0,diff) if self.backend == "numpy" else torch.clamp(diff,min=0)
        )
    
    @property
    def pt(self):
        return self._op(np.sqrt, torch.sqrt, self.px**2 + self.py**2)
    
    @property
    def eta(self):
        p = self._op(np.sqrt,torch.sqrt, self.px**2 + self.py**2 + self.pz**2)
        eps = 1e-12 if self.backend == "numpy" else torch.tensor(1e-12,device=self.px.device)
        num = p + self.pz
        den = p - self.pz + eps
        return 0.5 * self._op(np.log,torch.log, num/den)
    
    @property
    def phi(self):
        return self._op(np.arctan2,torch.atan2,self.py,self.px)
    
    def dot(self,other):
        return self.E * other.E - (self.px * other.px + self.py * other.py + self.pz * other.pz)
    
    def to(self,device):
        if self.backend != "torch":
            raise TypeError("Only torch tensors can be moved to devices!")
        return Vec4(
            self.E.to(device),
            self.px.to(device),
            self.py.to(device),
            self.pz.to(device)
        )
    
    def __repr__(self):
        return f"FourVector(E={self.E}, px={self.px}, py={self.py}, pz={self.pz})"

    def __add__(self,other):
        return Vec4(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        ) 