from abc import ABC, abstractmethod

class DemoABC(ABC):

    @abstractmethod
    def getLogTextAndProcessedFrame(self, frame):
        pass
    
    @abstractmethod
    def getLogText(self, frame):
        pass