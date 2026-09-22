"""Evaluator registry used by the grading entry points."""
from .base import Evaluator, ASR
from .ollama import Ollama, GptOss20b
from .refusal import Refusal, LocalRefusal
from .llamaguard import LlamaGuard4

EVALUATORS = {"llamaguard4": LlamaGuard4, "gpt-oss-20b": GptOss20b,
              "xstest-refusal": Refusal}
