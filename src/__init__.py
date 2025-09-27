"""
Methane Emissions Mapping and Risk Analysis Package

A comprehensive spatial analysis workflow for methane emissions risk assessment
in Grand Teton National Park using LiDAR, satellite data, and infrastructure datasets.
"""

__version__ = "1.0.0"
__author__ = "Spatial Analysis Team"
__email__ = "spatial.analysis@example.com"

from .data_processing import LiDARProcessor, SatelliteDataProcessor, InfrastructureProcessor
from .spatial_analysis import RiskAnalyzer, ProximityAnalyzer, TemporalAnalyzer
from .visualization import RiskMapper, StatisticalPlotter, InteractiveMapper
from .utils import ConfigManager, DataValidator, Logger

__all__ = [
    "LiDARProcessor",
    "SatelliteDataProcessor", 
    "InfrastructureProcessor",
    "RiskAnalyzer",
    "ProximityAnalyzer", 
    "TemporalAnalyzer",
    "RiskMapper",
    "StatisticalPlotter",
    "InteractiveMapper",
    "ConfigManager",
    "DataValidator",
    "Logger"
]