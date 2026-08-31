"""Prompting strategies.

Importing this package registers every strategy, so `available()` reflects what
is installed without any central list to keep in sync.
"""
from .base import Context, Strategy, StrategyResult, available, get_strategy, register
from . import zero_shot, few_shot, cot, tot, got   # noqa: F401  (registration)

__all__ = ["Context", "Strategy", "StrategyResult", "available", "get_strategy",
           "register"]
