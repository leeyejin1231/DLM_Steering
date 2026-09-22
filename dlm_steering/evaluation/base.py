"""Evaluator interfaces and resource lifetime."""
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Grades items (each a dict with 'prompt' and 'response')."""

    name: str

    @abstractmethod
    def evaluate(self, items, output_path=None):
        """Return items with grading fields added.

        output_path, if given, is a JSONL file: already-graded responses are
        skipped (resume) and new results are appended as they finish.
        """

    @abstractmethod
    def summarize(self, items):
        """Headline metrics over evaluate()'d items."""

    def close(self):
        """Release models / containers. Default: nothing held."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ASR(Evaluator):
    """Attack-success-rate grader: summarize() reports 'asr'."""
