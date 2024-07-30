from .abstract_pce import AbstractPCE
from math import factorial as fact
import torch
from .config import *

class PCE(AbstractPCE):
    def __init__(self, distributions, p):
        self.distributions = distributions
        self.p = p
        self.P = fact(self.vars + self.p) // (fact(self.vars) * fact(self.p))
        self.pce_coeffs = torch.rand(self.P) * 2 - 1

    @property
    def vars(self):
        return len(self.distributions)
    
    @property
    def degrees_sets(self):
        return zip(self.pce_coeffs, self.generate_degrees(self.vars, self.p))
    
    def polynom_coeffs(self, var, degree):
        return self.distributions[var].polynom_coeffs(degree)
    
    def linear_transform_coeffs(self, var):
        return self.distributions[var].linear_transform_coeffs
    
    @staticmethod
    def generate_degrees(m, p):
        def degrees_with_sum(m, s):
            if m == 1:
                yield [s]
                return
            if s == 0:
                yield [0] * m
                return
            for i in range(s+1):
                for right_part in degrees_with_sum(m-1, i):
                    yield [s-i] + right_part
        
        for s in range(p, -1, -1):
            yield from degrees_with_sum(m, s)
        return
