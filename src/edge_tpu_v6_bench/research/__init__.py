"""
Research framework for Edge TPU v6 comparative studies and academic publication
"""

from .baseline_framework import BaselineComparisonFramework
from .statistical_testing import StatisticalTestSuite
from .experimental_design import ExperimentalDesign
from .publication_tools import PublicationDataGenerator

__all__ = [
    'BaselineComparisonFramework',
    'StatisticalTestSuite', 
    'ExperimentalDesign',
    'PublicationDataGenerator'
]