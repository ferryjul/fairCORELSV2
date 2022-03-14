from __future__ import print_function, division, with_statement
from .corels import FairCorelsClassifierV2
from .utils import load_from_csv, RuleList
from .metrics import ConfusionMatrix, Metric

__version__ = "1.1"

__all__ = ["FairCorelsClassifierV2", "load_from_csv", "RuleList", "ConfusionMatrix", "Metric"]