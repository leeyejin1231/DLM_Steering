from abc import ABC, abstractmethod

class Defender(ABC):
    @abstractmethod
    def defend(self):
        pass

class Ours(Defender):
    def defend(self):
        pass

class DiffuGuard(Defender):
    def defend(self):
        pass

class SelfReminder(Defender):
    def defend(self):
        pass