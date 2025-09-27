"""
Data processing modules for the methane emissions analysis workflow.

This package contains processors for various data types including:
- LiDAR point cloud data
- Oil & gas infrastructure data
- Satellite-derived methane concentrations
- Environmental data layers
"""

from .lidar_processor import LiDARProcessor
from .infrastructure_processor import InfrastructureProcessor
from .satellite_processor import SatelliteDataProcessor

__all__ = [
    'LiDARProcessor',
    'InfrastructureProcessor', 
    'SatelliteDataProcessor'
]