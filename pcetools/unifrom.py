from scipy.special import legendre
from .distribution import Distribution
import torch

class Uniform(Distribution):
    def __init__(self, a, b):
        """
        Инициализация равномерного распределения на отрезке [a, b].
        :param a: Нижний предел интервала.
        :param b: Верхний предел интервала.
        """
        self.a = a
        self.b = b

    def polynom_coeffs(self, degree):
        return torch.tensor(legendre(degree).coeffs, dtype=torch.float32)
    
    @property
    def linear_transform_coeffs(self):
        return (2 / (self.b - self.a), -(self.b + self.a) / (self.b - self.a))