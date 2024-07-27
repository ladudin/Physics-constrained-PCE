from scipy.special import eval_hermite
from .distribution import Distribution

class Normal(Distribution):
    def __init__(self, mean, std_dev):
        """
        Инициализация нормального распределения с заданными параметрами.
        :param mean: Среднее значение.
        :param std_dev: Стандартное отклонение.
        """
        self.mean = mean
        self.std_dev = std_dev

    def polynom(self, x, degree):
        """
        Вычисление значения полинома Хермита степени `degree` в точке `x`.
        :param x: Точка, в которой вычисляется полином.
        :param degree: Степень полинома.
        :return: Значение полинома в точке `x`.
        """
        normalized_x = (x - self.mean) / self.std_dev
        return eval_hermite(degree, normalized_x)
