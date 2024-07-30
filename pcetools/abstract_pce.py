from abc import ABC, abstractmethod
import torch

class AbstractPCE(ABC):
    @property
    @abstractmethod
    def vars(self):
        pass

    @property
    @abstractmethod
    def degrees_sets(self):
        pass

    @abstractmethod
    def polynom_coeffs(self, var, degree):
        pass

    @abstractmethod
    def linear_transform_coeffs(self, var):
        return torch.ones(2)

    def transform(self, var, x):
        a, b = self.linear_transform_coeffs(var)
        return x * a + b

    def polynom(self, var, degree, x):
        polynom_coeffs = self.polynom_coeffs(var, degree)
        assert len(polynom_coeffs) == degree + 1

        p_features = torch.stack(
            [
                self.transform(var, x) ** (degree - j) 
             for j in range(degree+1)
             ], dim=1)

        return p_features @ polynom_coeffs
    
    def __call__(self, X):
        s = 0
        for pce_coeff, degrees in self.degrees_sets:
            phi = torch.stack(
                [
                    self.polynom(var_num, degrees[var_num], X[:, var_num]) 
                    for var_num in range(self.vars)
                ], dim=1)
            s += phi.prod(dim=-1) * pce_coeff
        return s
    
    def derivative(self, var):
        return Derivative(self, var)


class Derivative(AbstractPCE):
    def __init__(self, pce, dvar):
        self.pce = pce
        self.dvar = dvar

    @property
    def vars(self):
        return self.pce.vars

    @property
    def degrees_sets(self):
        for pce_coeff, degrees in self.pce.degrees_sets:
            if degrees[self.dvar]:
                degrees = list(degrees)
                degrees[self.dvar] -= 1
                yield pce_coeff, degrees
            yield 0.0, degrees
    
    def polynom_coeffs(self, var, degree):
        if var != self.dvar:
            return self.pce.polynom_coeffs(var, degree)

        coeffs = self.pce.polynom_coeffs(var, degree+1)
        assert len(coeffs) == degree + 2
        degrees = torch.arange(degree + 1, -1, -1)
        coeffs *= degrees
        coeffs *= self.pce.linear_transform_coeffs(var)[0]
        return coeffs[:-1] if len(coeffs) > 1 else coeffs

    def linear_transform_coeffs(self, var):
        return self.pce.linear_transform_coeffs(var)
