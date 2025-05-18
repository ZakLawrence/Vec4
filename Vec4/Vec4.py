import numpy as np 
import torch

class Vec4:
    def __init__(self,*x):
        if len(x) != 4:
            raise ValueError("4-vectors expect four values")
        
        self.x0 = self._ensure_array(x[0])
        self.x1 = self._ensure_array(x[1])
        self.x2 = self._ensure_array(x[2])
        self.x3 = self._ensure_array(x[3])

        if not (self.x0.shape == self.x1.shape == self.x2.shape == self.x3.shape):
            raise ValueError("All components must have the same shape")
        self.backend = self._detect_backend(self.x0)

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

    def __getitem__(self,idx):
        return self.__class__(
            self.x0[idx],
            self.x1[idx],
            self.x2[idx],
            self.x3[idx]
        )

    def __add__(self,other):
        if not isinstance(other,(Vec4,int,float)):
            raise TypeError(f"Unable to perform addition between 4-vectoro and {type(other)}!")
        if isinstance(other,Vec4):
            return Vec4(
                self.x0 + other.x0,
                self.x1 + other.x1,
                self.x2 + other.x2,
                self.x3 + other.x3
            )
        else:  
            return Vec4(
                self.x0 + other,
                self.x1 + other,
                self.x2 + other,
                self.x3 + other
            )

    def __iadd__(self,other):
        if not isinstance(other,(Vec4,int,float)):
            raise TypeError(f"Unable to perform addition between 4-vectoro and {type(other)}!")        
        if isinstance(other,Vec4):
            self.x0 += other.x0  
            self.x1 += other.x1              
            self.x2 += other.x2              
            self.x3 += other.x3              
            return self 
        else:
            self.x0 += other  
            self.x1 += other              
            self.x2 += other              
            self.x3 += other              
            return self 

    def __sub__(self,other):
        if not isinstance(other,(Vec4,int,float)):
            raise TypeError(f"Unable to perform addition between 4-vectoro and {type(other)}!")        
        if isinstance(other,Vec4):
            return Vec4(
                self.x0 - other.x0,
                self.x1 - other.x1,
                self.x2 - other.x2,
                self.x3 - other.x3
            )
        else:  
            return Vec4(
                self.x0 - other,
                self.x1 - other,
                self.x2 - other,
                self.x3 - other
            )
    
    def __isub__(self,other):
        if not isinstance(other,(Vec4,int,float)):
            raise TypeError(f"Unable to perform addition between 4-vectoro and {type(other)}!")        
        if isinstance(other,Vec4):
            self.x0 -= other.x0  
            self.x1 -= other.x1              
            self.x2 -= other.x2              
            self.x3 -= other.x3              
            return self 
        else:
            self.x0 -= other  
            self.x1 -= other              
            self.x2 -= other              
            self.x3 -= other              
            return self 
    
    def dot(self,other):
        if not isinstance(other,Vec4):
            raise TypeError("Dot product must be between two 4-vectors")
        return self.x0 * other.x0 - (self.x1 * other.x1 + self.x2 * other.x2 + self.x3 * other.x3)

    def __mul__(self,other):
        if isinstance(other,Vec4):
            return self.dot(other)
        elif isinstance(other,(int,float)):
            return Vec4(
                self.x0 * other,
                self.x1 * other,
                self.x2 * other,
                self.x3 * other
            )
        else: 
            raise TypeError(f"Unable to multiply 4-vector by {type(other)}")
        
    def __rmul__(self,other):
        return self * other
    
    def __imul__(self,other):
        if not isinstance(other,(int,float)):
            raise TypeError(f"Unable to perform multiplication by {type(other)}")
        self.x0 *= other
        self.x1 *= other
        self.x2 *= other
        self.x3 *= other
        return self
    
    def __neg__(self):
        return (-1) * self 
    
    def __truediv__(self,other):
        if not isinstance(other,(int,float)):
            raise TypeError(f"Unable to perform division by {type(other)}")
        return Vec4(
            self.x0 / other,
            self.x1 / other,
            self.x2 / other,
            self.x3 / other
        )
    
    def __floordiv__(self,other):
        if not isinstance(other,(int,float)):
            raise TypeError(f"Unable to perform division by {type(other)}")
        return Vec4(
            self.x0 // other,
            self.x1 // other,
            self.x2 // other,
            self.x3 // other
        )
    
    def __itruediv__(self,other):
        if not isinstance(other,(int,float)):
            raise TypeError(f"Unable to perform division by {type(other)}")
        self.x0 /= other
        self.x1 /= other
        self.x2 /= other
        self.x3 /= other
        return self
    
    def __ifloordiv__(self,other):
        if not isinstance(other,(int,float)):
            raise TypeError(f"Unable to perform division by {type(other)}")
        self.x0 //= other
        self.x1 //= other
        self.x2 //= other
        self.x3 //= other
        return self
    
    def __len__(self):
        return self.x0.shape[0]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    
    @property
    def mag2(self):
        return self * self
    
    @property
    def mag(self):
        return self._op(
            np.sqrt, torch.sqrt,
            self.mag2
        )

    @property
    def trans(self):
        return self._op(
            np.sqrt, torch.sqrt,
            self.x1**2 + self.x2**2
        )
    
    def to(self,device):
        if self.backend != "torch":
            raise TypeError("Only torch tensors can be moved to devices!")
        return Vec4(
            self.x0.to(device),
            self.x1.to(device),
            self.x2.to(device),
            self.x3.to(device)
        )
    
    def __repr__(self):
        return f"Vec4(x0={self.x0}, x1={self.x1}, x2={self.x2}, x3={self.x3})"

    @property
    def eta(self):
        p = self._op(
            np.sqrt,torch.sqrt, 
            self.x1**2 + self.x2**2 + self.x3**2
            )
        eps = 1e-12 if self.backend == "numpy" else torch.tensor(1e-12,device=self.x1.device)
        num = p + self.x3
        den = p - self.x3 + eps
        return 0.5 * self._op(np.log,torch.log, num/den)
    
    @property
    def phi(self):
        return self._op(np.arctan2,torch.atan2,self.x2,self.x1)
    
    @property
    def theta(self):
        return self._op(np.arctan2,torch.atan2,self.trans,self.x3)
    
    def DeltaR(self,other):
        if not isinstance(other,Vec4):
            raise TypeError("Must be of type Vec4")
        return self._op(
            np.sqrt,torch.sqrt,
            (self.eta - other.ets)**2 + (self.phi - other.phi)**2
            )
            
    #@property
    #def mass(self):
    #    E2 = self.E**2
    #    p2 = self.px**2 + self.py**2 + self.pz**2

    #    diff = E2 - p2
    #    return self._op(
    #        np.sqrt, torch.sqrt,
    #        np.maximum(0,diff) if self.backend == "numpy" else torch.clamp(diff,min=0)
    #    )
    
    #@property
    #def pt(self):
    #    return self._op(np.sqrt, torch.sqrt, self.px**2 + self.py**2)
    
    
    
