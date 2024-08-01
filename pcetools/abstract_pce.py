from abc import ABC, abstractmethod
import torch
from . import config

class AbstractPCE(ABC):
    def __init__(self, vars):
        self.vars = vars
        self.cache = dict()

    @property
    @abstractmethod
    def components(self):
        pass

    @abstractmethod
    def polynom_coeffs(self, var, degree):
        pass

    @abstractmethod
    def linear_transform_coeffs(self, var):
        pass

    def transform(self, var, x):
        a, b = self.linear_transform_coeffs(var)
        return x * a + b

    def polynom(self, var, degree, x):
        value = self.cache.get((var, degree), None)
        if value is not None:
            return value

        polynom_coeffs = self.polynom_coeffs(var, degree)
        assert len(polynom_coeffs) == degree + 1

        p_features = torch.stack(
            [
                self.transform(var, x) ** (degree - j) 
             for j in range(degree+1)
             ], dim=1)

        value = p_features @ polynom_coeffs
        self.cache[(var, degree)] = value
        return value
    
    def __call__(self, X):
        assert X.dtype == config.dtype, (
            f"Входные данные должны иметь тип {config.dtype}"
        )
        s = 0
        for pce_coeff, degrees in self.components:
            if pce_coeff == 0.0:
                continue
            phi = torch.stack(
                [
                    self.polynom(var_num, degrees[var_num], X[:, var_num]) 
                    for var_num in range(self.vars)
                ], dim=1)
            s += phi.prod(dim=-1) * pce_coeff
        self.cache.clear()
        return s
    
    def derivative(self, var):
        return Derivative(self, var)


class Derivative(AbstractPCE):
    def __init__(self, pce, dvar):
        super().__init__(pce.vars)
        self.pce = pce
        self.dvar = dvar

    @property
    def components(self):
        for pce_coeff, degrees in self.pce.components:
            if degrees[self.dvar]:
                degrees = list(degrees)
                degrees[self.dvar] -= 1
                yield pce_coeff, degrees
            else:
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
