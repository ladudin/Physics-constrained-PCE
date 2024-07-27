from abc import ABC, abstractmethod

class Distribution(ABC):
    @abstractmethod
    def __init__(self, *params):
        """
        Конструктор для инициализации параметров распределения.
        """
        pass

    @abstractmethod
    def polynom(self, x, degree):
        """
        Вычисление значения ортогонального полинома степени degree в точке x
        При необходимости значение x нормализуется в зависимости от параметров
        распределения.
        """
        pass
