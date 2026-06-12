from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, image=None, designation="", description=""):
        pass