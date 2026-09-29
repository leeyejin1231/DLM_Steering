from abc import ABC, abstractmethod


class Evaluator(ABC):

    name: str

    @abstractmethod
    def evaluate(self, items, output_path=None):
        ...

    @abstractmethod
    def summarize(self, items):
        ...

    def close(self):
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ASR(Evaluator):
    ...
