"""
Configuration management utilities for the methane emissions analysis workflow.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and validation for the analysis workflow."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'data_sources.lidar.resolution')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_study_area_bounds(self) -> tuple:
        """Get study area bounding box."""
        bounds = self.get('study_area.bounds.bbox')
        if not bounds or len(bounds) != 4:
            raise ValueError("Study area bounds must be [minx, miny, maxx, maxy]")
        return tuple(bounds)
    
    def get_epsg_code(self) -> int:
        """Get the EPSG code for the study area."""
        return self.get('study_area.epsg', 4326)
    
    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """
        Get configuration for a specific data source.
        
        Args:
            source: Data source name (e.g., 'lidar', 'oil_gas_infrastructure')
            
        Returns:
            Data source configuration
        """
        return self.get(f'data_sources.{source}', {})
    
    def get_analysis_params(self) -> Dict[str, Any]:
        """Get analysis parameters."""
        return self.get('analysis', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('outputs', {})
    
    def validate_config(self) -> bool:
        """
        Validate the configuration file for required fields.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        required_fields = [
            'study_area.bounds.bbox',
            'study_area.epsg',
            'data_sources',
            'analysis.risk_factors.weights',
            'outputs.formats'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                raise ValueError(f"Required configuration field missing: {field}")
        
        # Validate bounding box
        bbox = self.get_study_area_bounds()
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid bounding box: minx < maxx and miny < maxy required")
        
        # Validate risk factor weights sum to 1
        weights = self.get('analysis.risk_factors.weights', {})
        if weights:
            total_weight = sum(weights.values())
            if not (0.99 <= total_weight <= 1.01):  # Allow for small floating point errors
                raise ValueError(f"Risk factor weights must sum to 1.0, got {total_weight}")
        
        return True


class EnvironmentManager:
    """Manages environment variables and paths."""
    
    @staticmethod
    def get_data_dir() -> Path:
        """Get the data directory path."""
        return Path(os.getenv('METHANE_DATA_DIR', './data'))
    
    @staticmethod
    def get_temp_dir() -> Path:
        """Get the temporary directory path."""
        temp_dir = Path(os.getenv('METHANE_TEMP_DIR', './temp'))
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get the output directory path."""
        output_dir = Path(os.getenv('METHANE_OUTPUT_DIR', './data/outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @staticmethod
    def setup_directories():
        """Create necessary directories if they don't exist."""
        directories = [
            EnvironmentManager.get_data_dir(),
            EnvironmentManager.get_temp_dir(),
            EnvironmentManager.get_output_dir(),
            Path('./logs')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)