from abc import ABC, abstractmethod

class Attacker(ABC):
    @abstractmethod
    def generate(self):
        pass

class Prefix(Attacker):
    def generate(self):
        pass

class PAP(Attacker):
    def generate(self):
        pass

class PAIR(Attacker):
    def generate(self):
        pass

class DIJA(Attacker):
    def generate(self):
        pass
