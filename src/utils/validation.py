"""
Data validation utilities for the methane emissions analysis workflow.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, Polygon
import logging

from .logging_utils import Logger


class DataValidator:
    """Validates input data for the methane emissions analysis workflow."""
    
    def __init__(self):
        self.logger = Logger.get_logger(__name__)
    
    def validate_bounding_box(self, bbox: Union[List, Tuple]) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            bbox: Bounding box as [minx, miny, maxx, maxy]
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if len(bbox) != 4:
                self.logger.error(f"Bounding box must have 4 coordinates, got {len(bbox)}")
                return False
            
            minx, miny, maxx, maxy = bbox
            
            if minx >= maxx:
                self.logger.error(f"minx ({minx}) must be less than maxx ({maxx})")
                return False
                
            if miny >= maxy:
                self.logger.error(f"miny ({miny}) must be less than maxy ({maxy})")
                return False
            
            # Check if coordinates are reasonable (rough global bounds)
            if not (-180 <= minx <= 180) or not (-180 <= maxx <= 180):
                self.logger.error(f"Longitude values out of range: {minx}, {maxx}")
                return False
                
            if not (-90 <= miny <= 90) or not (-90 <= maxy <= 90):
                self.logger.error(f"Latitude values out of range: {miny}, {maxy}")
                return False
            
            return True
            
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error validating bounding box: {e}")
            return False
    
    def validate_geodataframe(self, gdf: gpd.GeoDataFrame, 
                            required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate a GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(gdf, gpd.GeoDataFrame):
                self.logger.error("Input is not a GeoDataFrame")
                return False
            
            if gdf.empty:
                self.logger.warning("GeoDataFrame is empty")
                return True
            
            # Check for geometry column
            if gdf.geometry.empty.all():
                self.logger.error("GeoDataFrame has no valid geometries")
                return False
            
            # Check CRS
            if gdf.crs is None:
                self.logger.warning("GeoDataFrame has no CRS defined")
            
            # Check for required columns
            if required_columns:
                missing_columns = set(required_columns) - set(gdf.columns)
                if missing_columns:
                    self.logger.error(f"Missing required columns: {missing_columns}")
                    return False
            
            # Check for invalid geometries
            invalid_geoms = ~gdf.geometry.is_valid
            if invalid_geoms.any():
                invalid_count = invalid_geoms.sum()
                self.logger.warning(f"Found {invalid_count} invalid geometries")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating GeoDataFrame: {e}")
            return False
    
    def validate_raster(self, raster_path: Union[str, Path]) -> bool:
        """
        Validate a raster file.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            raster_path = Path(raster_path)
            
            if not raster_path.exists():
                self.logger.error(f"Raster file does not exist: {raster_path}")
                return False
            
            with rasterio.open(raster_path) as src:
                # Check basic properties
                if src.width == 0 or src.height == 0:
                    self.logger.error(f"Raster has zero dimensions: {src.width}x{src.height}")
                    return False
                
                if src.count == 0:
                    self.logger.error("Raster has no bands")
                    return False
                
                # Check CRS
                if src.crs is None:
                    self.logger.warning("Raster has no CRS defined")
                
                # Check for reasonable bounds
                bounds = src.bounds
                if not self.validate_bounding_box([bounds.left, bounds.bottom, 
                                                 bounds.right, bounds.top]):
                    return False
                
                # Sample data to check for issues
                try:
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, 
                                                                       min(100, src.width),
                                                                       min(100, src.height)))
                    if np.all(np.isnan(sample)):
                        self.logger.warning("Raster sample contains only NaN values")
                
                except Exception as e:
                    self.logger.error(f"Error reading raster sample: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating raster: {e}")
            return False
    
    def validate_point_cloud(self, pc_path: Union[str, Path]) -> bool:
        """
        Validate a point cloud file (LAS/LAZ).
        
        Args:
            pc_path: Path to point cloud file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            import laspy
            
            pc_path = Path(pc_path)
            
            if not pc_path.exists():
                self.logger.error(f"Point cloud file does not exist: {pc_path}")
                return False
            
            with laspy.open(pc_path) as las_file:
                header = las_file.header
                
                # Check point count
                if header.point_count == 0:
                    self.logger.error("Point cloud file contains no points")
                    return False
                
                # Check bounds
                bounds = [header.x_min, header.y_min, header.x_max, header.y_max]
                if not self.validate_bounding_box(bounds):
                    return False
                
                # Check if we can read a sample of points
                try:
                    sample_size = min(1000, header.point_count)
                    points = las_file.read().points[:sample_size]
                    
                    if len(points) == 0:
                        self.logger.error("Cannot read points from file")
                        return False
                
                except Exception as e:
                    self.logger.error(f"Error reading point cloud sample: {e}")
                    return False
            
            return True
            
        except ImportError:
            self.logger.error("laspy library not available for point cloud validation")
            return False
        except Exception as e:
            self.logger.error(f"Error validating point cloud: {e}")
            return False
    
    def validate_coordinate_reference_system(self, crs: Any) -> bool:
        """
        Validate a coordinate reference system.
        
        Args:
            crs: CRS object or EPSG code
            
        Returns:
            True if valid, False otherwise
        """
        try:
            import pyproj
            
            if isinstance(crs, int):
                # EPSG code
                try:
                    pyproj.CRS.from_epsg(crs)
                    return True
                except pyproj.exceptions.CRSError:
                    self.logger.error(f"Invalid EPSG code: {crs}")
                    return False
            
            elif isinstance(crs, str):
                # CRS string
                try:
                    pyproj.CRS.from_string(crs)
                    return True
                except pyproj.exceptions.CRSError:
                    self.logger.error(f"Invalid CRS string: {crs}")
                    return False
            
            elif hasattr(crs, 'to_epsg'):
                # CRS object
                return True
            
            else:
                self.logger.error(f"Unrecognized CRS type: {type(crs)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating CRS: {e}")
            return False
    
    def validate_data_consistency(self, datasets: Dict[str, Any]) -> bool:
        """
        Validate consistency between multiple datasets.
        
        Args:
            datasets: Dictionary of datasets to validate
            
        Returns:
            True if consistent, False otherwise
        """
        try:
            bounds_list = []
            crs_list = []
            
            for name, data in datasets.items():
                if hasattr(data, 'bounds'):
                    bounds_list.append((name, data.bounds))
                
                if hasattr(data, 'crs'):
                    crs_list.append((name, data.crs))
            
            # Check spatial overlap
            if len(bounds_list) > 1:
                ref_name, ref_bounds = bounds_list[0]
                ref_poly = Polygon([
                    (ref_bounds[0], ref_bounds[1]),
                    (ref_bounds[2], ref_bounds[1]),
                    (ref_bounds[2], ref_bounds[3]),
                    (ref_bounds[0], ref_bounds[3])
                ])
                
                for name, bounds in bounds_list[1:]:
                    poly = Polygon([
                        (bounds[0], bounds[1]),
                        (bounds[2], bounds[1]),
                        (bounds[2], bounds[3]),
                        (bounds[0], bounds[3])
                    ])
                    
                    if not ref_poly.intersects(poly):
                        self.logger.warning(f"No spatial overlap between {ref_name} and {name}")
            
            # Check CRS consistency
            if len(crs_list) > 1:
                ref_crs = crs_list[0][1]
                for name, crs in crs_list[1:]:
                    if crs != ref_crs:
                        self.logger.info(f"CRS mismatch: {crs_list[0][0]} has {ref_crs}, {name} has {crs}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data consistency: {e}")
            return False