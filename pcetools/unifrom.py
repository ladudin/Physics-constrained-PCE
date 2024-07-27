from scipy.special import eval_legendre
from .distribution import Distribution

class Uniform(Distribution):
    def __init__(self, a, b):
        """
        Инициализация равномерного распределения на отрезке [a, b].
        :param a: Нижний предел интервала.
        :param b: Верхний предел интервала.
        """
        self.a = a
        self.b = b

    def polynom(self, x, degree):
        """
        Вычисление значения полинома Лежандра степени `degree` в точке `x`.
        :param x: Точка, в которой вычисляется полином.
        :param degree: Степень полинома.
        :return: Значение полинома в точке `x`.
        """
        normalized_x = 2 * (x - self.a) / (self.b - self.a) - 1
        return eval_legendre(degree, normalized_x)
