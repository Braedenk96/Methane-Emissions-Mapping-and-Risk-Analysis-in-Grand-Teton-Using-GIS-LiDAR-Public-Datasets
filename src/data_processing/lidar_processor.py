"""
LiDAR data processing module for methane emissions analysis.

This module handles the processing of LiDAR point cloud data to extract
terrain and surface features relevant to methane emissions modeling.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import rasterio
from rasterio.transform import from_bounds
import laspy
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN

from ..utils import Logger, log_function_call, ProgressLogger, DataValidator


class LiDARProcessor:
    """Processes LiDAR point cloud data for methane emissions analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize LiDAR processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.validator = DataValidator()
        
        # Processing parameters
        self.ground_threshold = config.get('ground_threshold', 2.0)  # meters
        self.vegetation_height_min = config.get('vegetation_height_min', 0.5)  # meters
        self.grid_resolution = config.get('grid_resolution', 1.0)  # meters
        
    @log_function_call
    def load_point_cloud(self, las_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load LiDAR point cloud from LAS/LAZ file.
        
        Args:
            las_path: Path to LAS/LAZ file
            
        Returns:
            DataFrame with point cloud data
        """
        las_path = Path(las_path)
        
        if not self.validator.validate_point_cloud(las_path):
            raise ValueError(f"Invalid point cloud file: {las_path}")
        
        self.logger.info(f"Loading point cloud from {las_path}")
        
        with laspy.open(las_path) as las_file:
            las = las_file.read()
            
            # Extract point coordinates and attributes
            points_data = {
                'x': las.x,
                'y': las.y,
                'z': las.z,
                'intensity': las.intensity if hasattr(las, 'intensity') else np.zeros(len(las.x)),
                'classification': las.classification if hasattr(las, 'classification') else np.zeros(len(las.x)),
                'return_number': las.return_number if hasattr(las, 'return_number') else np.ones(len(las.x)),
                'number_of_returns': las.number_of_returns if hasattr(las, 'number_of_returns') else np.ones(len(las.x))
            }
            
            # Add RGB if available
            if hasattr(las, 'red'):
                points_data['red'] = las.red
                points_data['green'] = las.green
                points_data['blue'] = las.blue
        
        df = pd.DataFrame(points_data)
        
        self.logger.info(f"Loaded {len(df)} points from {las_path}")
        return df
    
    @log_function_call
    def classify_ground_points(self, points: pd.DataFrame, 
                              method: str = 'cloth_simulation') -> pd.DataFrame:
        """
        Classify ground points using various algorithms.
        
        Args:
            points: Point cloud DataFrame
            method: Ground classification method ('cloth_simulation', 'progressive_morphology', 'slope_based')
            
        Returns:
            DataFrame with ground classification
        """
        self.logger.info(f"Classifying ground points using {method} method")
        
        points_copy = points.copy()
        
        if method == 'slope_based':
            # Simple slope-based ground classification
            ground_mask = self._classify_ground_slope_based(points)
        elif method == 'progressive_morphology':
            # Progressive morphological filtering
            ground_mask = self._classify_ground_progressive_morphology(points)
        else:
            # Default to slope-based for simplicity
            ground_mask = self._classify_ground_slope_based(points)
        
        points_copy['is_ground'] = ground_mask
        
        ground_count = ground_mask.sum()
        self.logger.info(f"Classified {ground_count} ground points ({ground_count/len(points)*100:.1f}%)")
        
        return points_copy
    
    def _classify_ground_slope_based(self, points: pd.DataFrame) -> np.ndarray:
        """
        Simple slope-based ground classification.
        
        Args:
            points: Point cloud DataFrame
            
        Returns:
            Boolean mask for ground points
        """
        # Grid the area and find lowest points in each cell
        grid_size = 5.0  # meters
        
        x_min, x_max = points['x'].min(), points['x'].max()
        y_min, y_max = points['y'].min(), points['y'].max()
        
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Digitize points into grid cells
        x_indices = np.digitize(points['x'], x_bins)
        y_indices = np.digitize(points['y'], y_bins)
        
        ground_mask = np.zeros(len(points), dtype=bool)
        
        # For each grid cell, mark lowest points as potential ground
        for x_idx in range(1, len(x_bins)):
            for y_idx in range(1, len(y_bins)):
                cell_mask = (x_indices == x_idx) & (y_indices == y_idx)
                if cell_mask.any():
                    cell_points = points[cell_mask]
                    # Find lowest points within threshold
                    min_z = cell_points['z'].min()
                    ground_candidates = cell_mask & (points['z'] <= min_z + self.ground_threshold)
                    ground_mask |= ground_candidates
        
        return ground_mask
    
    def _classify_ground_progressive_morphology(self, points: pd.DataFrame) -> np.ndarray:
        """
        Progressive morphological filtering for ground classification.
        
        Args:
            points: Point cloud DataFrame
            
        Returns:
            Boolean mask for ground points
        """
        # Simplified implementation - in practice would use more sophisticated algorithm
        return self._classify_ground_slope_based(points)
    
    @log_function_call
    def generate_dtm(self, points: pd.DataFrame, resolution: float = 1.0) -> Tuple[np.ndarray, rasterio.transform.Affine, Dict]:
        """
        Generate Digital Terrain Model from ground points.
        
        Args:
            points: Point cloud DataFrame with ground classification
            resolution: Grid resolution in meters
            
        Returns:
            Tuple of (DTM array, transform, metadata)
        """
        self.logger.info(f"Generating DTM with {resolution}m resolution")
        
        # Filter to ground points
        ground_points = points[points.get('is_ground', True)].copy()
        
        if len(ground_points) == 0:
            raise ValueError("No ground points available for DTM generation")
        
        # Define grid bounds
        x_min, x_max = ground_points['x'].min(), ground_points['x'].max()
        y_min, y_max = ground_points['y'].min(), ground_points['y'].max()
        
        # Add small buffer
        buffer = resolution * 2
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer
        
        # Create coordinate grids
        x_coords = np.arange(x_min, x_max + resolution, resolution)
        y_coords = np.arange(y_max, y_min - resolution, -resolution)  # Top to bottom
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Interpolate elevation values
        self.logger.info("Interpolating elevation values...")
        points_xyz = ground_points[['x', 'y', 'z']].values
        
        # Use griddata for interpolation
        dtm = griddata(
            points_xyz[:, :2],  # x, y coordinates
            points_xyz[:, 2],   # z values
            (xx, yy),
            method='linear',
            fill_value=np.nan
        )
        
        # Create transform
        transform = from_bounds(x_min, y_min, x_max, y_max, dtm.shape[1], dtm.shape[0])
        
        # Metadata
        metadata = {
            'driver': 'GTiff',
            'height': dtm.shape[0],
            'width': dtm.shape[1],
            'count': 1,
            'dtype': dtm.dtype,
            'transform': transform,
            'nodata': np.nan
        }
        
        self.logger.info(f"Generated DTM with shape {dtm.shape}")
        return dtm, transform, metadata
    
    @log_function_call
    def generate_dsm(self, points: pd.DataFrame, resolution: float = 1.0) -> Tuple[np.ndarray, rasterio.transform.Affine, Dict]:
        """
        Generate Digital Surface Model from all points.
        
        Args:
            points: Point cloud DataFrame
            resolution: Grid resolution in meters
            
        Returns:
            Tuple of (DSM array, transform, metadata)
        """
        self.logger.info(f"Generating DSM with {resolution}m resolution")
        
        # Define grid bounds
        x_min, x_max = points['x'].min(), points['x'].max()
        y_min, y_max = points['y'].min(), points['y'].max()
        
        # Add small buffer
        buffer = resolution * 2
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer
        
        # Create coordinate grids
        x_coords = np.arange(x_min, x_max + resolution, resolution)
        y_coords = np.arange(y_max, y_min - resolution, -resolution)
        
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # For DSM, we want the maximum elevation in each cell
        x_indices = np.digitize(points['x'], x_coords)
        y_indices = np.digitize(points['y'], y_coords)
        
        dsm = np.full((len(y_coords), len(x_coords)), np.nan)
        
        progress = ProgressLogger(len(points), "DSM Generation")
        
        for i, (x_idx, y_idx, z) in enumerate(zip(x_indices, y_indices, points['z'])):
            if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                current_val = dsm[y_idx, x_idx]
                if np.isnan(current_val) or z > current_val:
                    dsm[y_idx, x_idx] = z
            
            if i % 10000 == 0:
                progress.update(10000)
        
        progress.complete()
        
        # Create transform
        transform = from_bounds(x_min, y_min, x_max, y_max, dsm.shape[1], dsm.shape[0])
        
        # Metadata
        metadata = {
            'driver': 'GTiff',
            'height': dsm.shape[0],
            'width': dsm.shape[1],
            'count': 1,
            'dtype': dsm.dtype,
            'transform': transform,
            'nodata': np.nan
        }
        
        self.logger.info(f"Generated DSM with shape {dsm.shape}")
        return dsm, transform, metadata
    
    @log_function_call
    def calculate_vegetation_metrics(self, points: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate vegetation metrics from LiDAR data.
        
        Args:
            points: Point cloud DataFrame with ground classification
            
        Returns:
            DataFrame with vegetation metrics per grid cell
        """
        self.logger.info("Calculating vegetation metrics")
        
        # Ensure ground classification exists
        if 'is_ground' not in points.columns:
            points = self.classify_ground_points(points)
        
        # Calculate height above ground
        # This is simplified - in practice would use more sophisticated normalization
        ground_points = points[points['is_ground']]
        if len(ground_points) == 0:
            self.logger.warning("No ground points available for height normalization")
            points['height_above_ground'] = points['z']
        else:
            # Simple approach: subtract minimum ground elevation
            min_ground_z = ground_points['z'].min()
            points['height_above_ground'] = points['z'] - min_ground_z
        
        # Grid-based vegetation metrics
        grid_size = 10.0  # meters
        
        x_min, x_max = points['x'].min(), points['x'].max()
        y_min, y_max = points['y'].min(), points['y'].max()
        
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        metrics = []
        
        for i, x_center in enumerate(x_bins[:-1]):
            for j, y_center in enumerate(y_bins[:-1]):
                # Define grid cell bounds
                x_left = x_center
                x_right = x_bins[i + 1]
                y_bottom = y_center
                y_top = y_bins[j + 1]
                
                # Filter points in this cell
                cell_mask = (
                    (points['x'] >= x_left) & (points['x'] < x_right) &
                    (points['y'] >= y_bottom) & (points['y'] < y_top)
                )
                cell_points = points[cell_mask]
                
                if len(cell_points) == 0:
                    continue
                
                # Calculate vegetation metrics
                vegetation_points = cell_points[
                    cell_points['height_above_ground'] >= self.vegetation_height_min
                ]
                
                metrics.append({
                    'x_center': (x_left + x_right) / 2,
                    'y_center': (y_bottom + y_top) / 2,
                    'total_points': len(cell_points),
                    'vegetation_points': len(vegetation_points),
                    'vegetation_cover': len(vegetation_points) / len(cell_points) if len(cell_points) > 0 else 0,
                    'max_height': cell_points['height_above_ground'].max(),
                    'mean_height': cell_points['height_above_ground'].mean(),
                    'height_std': cell_points['height_above_ground'].std(),
                    'canopy_density': len(vegetation_points) / (grid_size ** 2)  # points per m²
                })
        
        vegetation_df = pd.DataFrame(metrics)
        self.logger.info(f"Calculated vegetation metrics for {len(vegetation_df)} grid cells")
        
        return vegetation_df
    
    @log_function_call
    def save_raster(self, array: np.ndarray, transform: rasterio.transform.Affine, 
                   metadata: Dict, output_path: Union[str, Path], crs: str = 'EPSG:32612'):
        """
        Save raster array to file.
        
        Args:
            array: Raster array
            transform: Affine transform
            metadata: Raster metadata
            output_path: Output file path
            crs: Coordinate reference system
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_copy = metadata.copy()
        metadata_copy['crs'] = crs
        
        with rasterio.open(output_path, 'w', **metadata_copy) as dst:
            dst.write(array, 1)
        
        self.logger.info(f"Saved raster to {output_path}")
    
    @log_function_call
    def process_lidar_tile(self, las_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Process a single LiDAR tile and generate all derived products.
        
        Args:
            las_path: Path to LAS/LAZ file
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping product names to file paths
        """
        las_path = Path(las_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing LiDAR tile: {las_path}")
        
        # Load point cloud
        points = self.load_point_cloud(las_path)
        
        # Classify ground points
        points = self.classify_ground_points(points)
        
        # Generate DTM
        dtm, dtm_transform, dtm_metadata = self.generate_dtm(points, self.grid_resolution)
        dtm_path = output_dir / f"{las_path.stem}_dtm.tif"
        self.save_raster(dtm, dtm_transform, dtm_metadata, dtm_path)
        
        # Generate DSM
        dsm, dsm_transform, dsm_metadata = self.generate_dsm(points, self.grid_resolution)
        dsm_path = output_dir / f"{las_path.stem}_dsm.tif"
        self.save_raster(dsm, dsm_transform, dsm_metadata, dsm_path)
        
        # Calculate CHM (Canopy Height Model)
        chm = dsm - dtm
        chm_path = output_dir / f"{las_path.stem}_chm.tif"
        self.save_raster(chm, dsm_transform, dsm_metadata, chm_path)
        
        # Calculate vegetation metrics
        vegetation_metrics = self.calculate_vegetation_metrics(points)
        vegetation_path = output_dir / f"{las_path.stem}_vegetation_metrics.csv"
        vegetation_metrics.to_csv(vegetation_path, index=False)
        
        results = {
            'dtm': dtm_path,
            'dsm': dsm_path,
            'chm': chm_path,
            'vegetation_metrics': vegetation_path,
            'point_count': len(points),
            'ground_point_count': points['is_ground'].sum()
        }
        
        self.logger.info(f"Completed processing LiDAR tile: {las_path}")
        return results