from scipy.special import eval_laguerre
from .distribution import Distribution

class Exponential(Distribution):
    def __init__(self, rate):
        """
        Инициализация экспоненциального распределения с заданным параметром rate.
        :param rate: Параметр распределения.
        """
        self.rate = rate

    def polynom(self, x, degree):
        """
        Вычисление значения полинома Лагерра степени `degree` в точке `x`.
        :param x: Точка, в которой вычисляется полином.
        :param degree: Степень полинома.
        :return: Значение полинома в точке `x`.
        """
        return eval_laguerre(degree, x / self.rate)
