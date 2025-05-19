from .Vec4 import Vec4
import numpy as np
import torch
from .utils import ensure_array, detect_backend, op

class MomentumVec4(Vec4):
    @staticmethod
    def m_eta_phi_pt(m, eta, phi, pt):
        m   = ensure_array(m) 
        eta = ensure_array(eta) 
        phi = ensure_array(phi) 
        pt  = ensure_array(pt) 

      
        px = op(np.cos,torch.cos,phi) * pt
        py = op(np.sin,torch.sin,phi) * pt
        exp_eta = op(np.exp,torch.exp, -eta)
        theta = 2 * op(
           np.arctan, torch.atan,
           exp_eta
        )
        pz = pt / op(np.tan,torch.tan,theta)
        e = op(np.sqrt,torch.sqrt, m**2 + px**2 + py**2 + pz**2)

        return MomentumVec4(e, px, py, pz)
    
    @property
    def e(self):
        return self.x0
    
    @property 
    def P2(self):
        return self.x1**2 + self.x2**2 + self.x3**2    

    @property
    def P(self):
        return op(np.sqrt,torch.sqrt,self.P2)
    
    @property
    def px(self):
        return self.x1
    
    @property
    def py(self):
        return self.x2
    
    @property
    def pz(self):
        return self.x3
    
    @property
    def pt(self):
        return self.trans
    
    @property
    def mass(self):
        E2 = self.e**2
        p2 = self.P2

        diff = E2 - p2
        return op(
            np.sqrt, torch.sqrt,
            np.maximum(0,diff) if self.backend == "numpy" else torch.clamp(diff,min=0)
        )
    
    def __repr__(self):
        return f"MomentumVec4(E={self.x0}, px={self.x1}, py={self.x2}, pz={self.x3})"
    
    
    