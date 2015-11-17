import equations
from .graph import CausalGraph, Equation
from .measure import MeasureCause, MeasureSuccess
from .payoff import PayoffMatrix
from .probability import (
    NS,
    Variable,
    make_variables,
    UniformDist,
    JointDist,
    JointDistByState
)

__version__ = "1.1.0"

__title__ = "causalinfo"
__description__ = "Information Measures on Causal Graphs."
__uri__ = "https://github/brettc/causalinfo/"

__author__ = "Brett Calcott"
__email__ = "brett.calcott@gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2015 Brett Calcott"

__all__ = [
    "CausalGraph",
    "Equation",
    "NS",
    "Variable",
    "make_variables",
    "UniformDist",
    "JointDist",
    "JointDistByState",
    "MeasureCause",
    "MeasureSuccess",
    "PayoffMatrix",
    "equations",
]
