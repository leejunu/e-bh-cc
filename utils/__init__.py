# this submodule contains utility functions and classes for the multiple testing problem, but are (mostly)
# agnostic to the distributional assumptions unique to each testing scenario. for example, multiple
# testing procedures, rejection procedures, calibration budgets, and other hyperparameters for CC will
# be defined here.

from .cc import CC
from .ci_sequence import hedged_cs
