"""Evaluation: is the retrieval-augmented pipeline actually working?"""
from .tamper import tamper
from .verdict import evaluate

__all__ = ["evaluate", "tamper"]
