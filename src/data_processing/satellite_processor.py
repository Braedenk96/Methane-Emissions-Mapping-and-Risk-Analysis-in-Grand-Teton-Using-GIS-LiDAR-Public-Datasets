"""
Satellite methane concentration data processing module.

This module handles the processing of satellite-derived methane concentration data
from sources like TROPOMI/Sentinel-5P for methane emissions analysis.
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import geopandas as gpd
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any
import requests
from datetime import datetime, timedelta
import warnings

from ..utils import Logger, log_function_call, ProgressLogger, DataValidator


class SatelliteDataProcessor:
    """Processes satellite-derived methane concentration data."""
    
    def __init__(self, config: Dict):
        """
        Initialize satellite data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.validator = DataValidator()
        
        # Processing parameters
        self.temporal_range = config.get('temporal_range', '2019-2024')
        self.spatial_resolution_km = config.get('resolution', '5.5x7km')
        
        # Methane concentration thresholds (ppb)
        self.background_ch4 = 1850  # Approximate background level
        self.anomaly_threshold = 50  # ppb above background
        self.high_anomaly_threshold = 100  # ppb above background
        
    @log_function_call
    def load_tropomi_data(self, data_source: Union[str, Path, Dict],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> xr.Dataset:
        """
        Load TROPOMI methane concentration data.
        
        Args:
            data_source: Path to NetCDF file, directory, or API configuration
            start_date: Start date for temporal filtering (YYYY-MM-DD)
            end_date: End date for temporal filtering (YYYY-MM-DD)
            
        Returns:
            xarray Dataset with methane concentration data
        """
        self.logger.info("Loading TROPOMI methane concentration data")
        
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            
            if data_path.is_file() and data_path.suffix == '.nc':
                # Single NetCDF file
                ds = xr.open_dataset(data_path)
            elif data_path.is_dir():
                # Directory with multiple NetCDF files
                nc_files = list(data_path.glob("*.nc"))
                if not nc_files:
                    raise FileNotFoundError(f"No NetCDF files found in {data_path}")
                
                datasets = []
                for nc_file in nc_files:
                    try:
                        ds = xr.open_dataset(nc_file)
                        datasets.append(ds)
                    except Exception as e:
                        self.logger.warning(f"Could not load {nc_file}: {e}")
                
                if not datasets:
                    raise ValueError("No valid NetCDF files could be loaded")
                
                # Concatenate along time dimension
                ds = xr.concat(datasets, dim='time')
            else:
                raise ValueError(f"Invalid data source: {data_path}")
        
        elif isinstance(data_source, dict):
            # Create sample data for demonstration
            ds = self._create_sample_tropomi_data(data_source)
        
        else:
            raise ValueError("data_source must be file path, directory, or dictionary")
        
        # Apply temporal filtering
        if start_date or end_date:
            ds = self._filter_temporal(ds, start_date, end_date)
        
        # Standardize dataset
        ds = self._standardize_tropomi_data(ds)
        
        self.logger.info(f"Loaded TROPOMI data with shape: {dict(ds.sizes)}")
        return ds
    
    def _create_sample_tropomi_data(self, bounds_config: Dict) -> xr.Dataset:
        """
        Create sample TROPOMI data for demonstration.
        
        Args:
            bounds_config: Dictionary with bounding box and temporal information
            
        Returns:
            Sample xarray Dataset
        """
        bbox = bounds_config.get('bbox', [-111.05, 43.65, -110.10, 43.85])
        minx, miny, maxx, maxy = bbox
        
        # Create spatial grid (approximate TROPOMI resolution ~5.5x7 km)
        lat_res = 0.05  # degrees (roughly 5.5 km)
        lon_res = 0.07  # degrees (roughly 7 km at this latitude)
        
        lats = np.arange(maxy, miny - lat_res, -lat_res)
        lons = np.arange(minx, maxx + lon_res, lon_res)
        
        # Create time series (daily data for one year)
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        times = pd.date_range(start_date, end_date, freq='D')
        
        # Create coordinate arrays
        time_coords = times
        lat_coords = lats
        lon_coords = lons
        
        # Generate realistic methane concentration data
        np.random.seed(42)
        
        # Base methane concentration (background level)
        base_ch4 = np.full((len(lats), len(lons)), self.background_ch4)
        
        # Add spatial patterns (higher concentrations near infrastructure)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Add some hotspots (simulating emissions sources)
        hotspot_locations = [
            (43.75, -110.8),  # Example hotspot 1
            (43.7, -110.5),   # Example hotspot 2
            (43.8, -110.3)    # Example hotspot 3
        ]
        
        spatial_anomalies = np.zeros_like(base_ch4)
        for hot_lat, hot_lon in hotspot_locations:
            # Gaussian distribution around hotspot
            distance = np.sqrt((lat_grid - hot_lat)**2 + (lon_grid - hot_lon)**2)
            hotspot_strength = 150 * np.exp(-distance**2 / (2 * 0.02**2))  # ~2 km standard deviation
            spatial_anomalies += hotspot_strength
        
        # Generate time series with seasonal and daily variations
        methane_data = np.zeros((len(times), len(lats), len(lons)))
        
        progress = ProgressLogger(len(times), "Generating sample methane data")
        
        for t, time in enumerate(times):
            # Seasonal variation (higher in winter)
            seasonal_factor = 1 + 0.1 * np.cos(2 * np.pi * (time.dayofyear - 30) / 365)
            
            # Random daily variation
            daily_variation = np.random.normal(0, 20, (len(lats), len(lons)))
            
            # Weather-related variation (wind dispersion effects)
            weather_factor = np.random.uniform(0.7, 1.3)
            
            # Combine all factors
            daily_ch4 = (base_ch4 + spatial_anomalies * weather_factor * seasonal_factor + daily_variation)
            
            # Ensure realistic values (no negative concentrations)
            daily_ch4 = np.maximum(daily_ch4, 1700)  # Minimum realistic CH4
            
            methane_data[t, :, :] = daily_ch4
            
            if t % 30 == 0:  # Update every 30 days
                progress.update(30)
        
        progress.complete()
        
        # Create quality flags and uncertainty estimates
        quality_flags = np.random.choice([0, 1], size=methane_data.shape, p=[0.2, 0.8])  # 80% good quality
        uncertainty = np.random.uniform(10, 50, size=methane_data.shape)  # Uncertainty in ppb
        
        # Create xarray Dataset
        ds = xr.Dataset({
            'methane_mixing_ratio': (['time', 'latitude', 'longitude'], methane_data),
            'qa_value': (['time', 'latitude', 'longitude'], quality_flags),
            'methane_mixing_ratio_precision': (['time', 'latitude', 'longitude'], uncertainty)
        }, coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords
        })
        
        # Add attributes
        ds.attrs = {
            'title': 'Sample TROPOMI Methane Data for Grand Teton Region',
            'source': 'Simulated data for demonstration',
            'institution': 'Methane Analysis Workflow',
            'units': 'ppb'
        }
        
        ds['methane_mixing_ratio'].attrs = {
            'long_name': 'Methane column-averaged dry-air mixing ratio',
            'units': 'ppb',
            'valid_range': [1700, 2500]
        }
        
        return ds
    
    def _filter_temporal(self, ds: xr.Dataset, start_date: Optional[str], end_date: Optional[str]) -> xr.Dataset:
        """
        Filter dataset by temporal range.
        
        Args:
            ds: Input dataset
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered dataset
        """
        if start_date:
            start_date = pd.Timestamp(start_date)
            ds = ds.sel(time=ds.time >= start_date)
        
        if end_date:
            end_date = pd.Timestamp(end_date)
            ds = ds.sel(time=ds.time <= end_date)
        
        return ds
    
    def _standardize_tropomi_data(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize TROPOMI dataset variable names and attributes.
        
        Args:
            ds: Raw TROPOMI dataset
            
        Returns:
            Standardized dataset
        """
        # Common variable name mappings
        var_mappings = {
            'ch4': 'methane_mixing_ratio',
            'CH4': 'methane_mixing_ratio',
            'methane': 'methane_mixing_ratio',
            'xch4': 'methane_mixing_ratio',
            'XCH4': 'methane_mixing_ratio'
        }
        
        # Rename variables if needed
        for old_name, new_name in var_mappings.items():
            if old_name in ds.variables and new_name not in ds.variables:
                ds = ds.rename({old_name: new_name})
        
        # Ensure required variables exist
        if 'methane_mixing_ratio' not in ds.variables:
            self.logger.warning("No methane concentration variable found in dataset")
            return ds
        
        # Apply quality filtering if quality flags are available
        quality_vars = ['qa_value', 'quality_flag', 'QA', 'qa']
        quality_var = None
        
        for qv in quality_vars:
            if qv in ds.variables:
                quality_var = qv
                break
        
        if quality_var:
            self.logger.info(f"Applying quality filtering using {quality_var}")
            # Typically, qa_value > 0.5 or quality_flag == 1 indicates good quality
            if quality_var == 'qa_value':
                good_quality = ds[quality_var] > 0.5
            else:
                good_quality = ds[quality_var] == 1
            
            # Set poor quality data to NaN
            ds['methane_mixing_ratio'] = ds['methane_mixing_ratio'].where(good_quality)
        
        # Ensure coordinates are correctly named
        coord_mappings = {
            'lat': 'latitude',
            'lon': 'longitude',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        
        for old_coord, new_coord in coord_mappings.items():
            if old_coord in ds.coords and new_coord not in ds.coords:
                ds = ds.rename({old_coord: new_coord})
        
        return ds
    
    @log_function_call
    def calculate_methane_anomalies(self, ds: xr.Dataset, 
                                  method: str = 'background_subtraction') -> xr.Dataset:
        """
        Calculate methane concentration anomalies.
        
        Args:
            ds: Input dataset with methane concentrations
            method: Anomaly calculation method ('background_subtraction', 'percentile', 'temporal_mean')
            
        Returns:
            Dataset with anomaly variables added
        """
        self.logger.info(f"Calculating methane anomalies using {method} method")
        
        ds_with_anomalies = ds.copy()
        
        if 'methane_mixing_ratio' not in ds.variables:
            raise ValueError("Dataset must contain 'methane_mixing_ratio' variable")
        
        ch4_data = ds['methane_mixing_ratio']
        
        if method == 'background_subtraction':
            # Subtract global background concentration
            anomaly = ch4_data - self.background_ch4
            
        elif method == 'percentile':
            # Use spatial percentiles to define background
            background_percentile = ch4_data.quantile(0.25, dim=['latitude', 'longitude'])
            anomaly = ch4_data - background_percentile
            
        elif method == 'temporal_mean':
            # Use temporal mean as background
            temporal_mean = ch4_data.mean(dim='time')
            anomaly = ch4_data - temporal_mean
            
        else:
            raise ValueError(f"Unknown anomaly method: {method}")
        
        # Add anomaly variables
        ds_with_anomalies['methane_anomaly'] = anomaly
        ds_with_anomalies['methane_anomaly'].attrs = {
            'long_name': 'Methane concentration anomaly',
            'units': 'ppb',
            'description': f'Calculated using {method} method'
        }
        
        # Create anomaly classifications
        anomaly_classes = xr.zeros_like(anomaly, dtype=int)
        anomaly_classes = anomaly_classes.where(~np.isnan(anomaly), -1)  # No data = -1
        anomaly_classes = anomaly_classes.where(anomaly < self.anomaly_threshold, 1)  # Low anomaly = 1
        anomaly_classes = anomaly_classes.where(anomaly < self.high_anomaly_threshold, 2)  # High anomaly = 2
        anomaly_classes = anomaly_classes.where(anomaly >= self.high_anomaly_threshold, 3)  # Very high = 3
        anomaly_classes = anomaly_classes.where(anomaly >= self.anomaly_threshold, 0)  # Background = 0
        
        ds_with_anomalies['anomaly_class'] = anomaly_classes
        ds_with_anomalies['anomaly_class'].attrs = {
            'long_name': 'Methane anomaly classification',
            'units': 'class',
            'description': '0=background, 1=low anomaly, 2=high anomaly, 3=very high anomaly, -1=no data'
        }
        
        self.logger.info("Completed methane anomaly calculations")
        return ds_with_anomalies
    
    @log_function_call
    def temporal_aggregation(self, ds: xr.Dataset, 
                           aggregation: str = 'monthly',
                           statistics: List[str] = ['mean', 'max', 'std']) -> xr.Dataset:
        """
        Perform temporal aggregation of methane data.
        
        Args:
            ds: Input dataset
            aggregation: Temporal aggregation period ('daily', 'weekly', 'monthly', 'seasonal', 'annual')
            statistics: List of statistics to calculate
            
        Returns:
            Temporally aggregated dataset
        """
        self.logger.info(f"Performing {aggregation} temporal aggregation")
        
        # Define aggregation periods
        if aggregation == 'daily':
            grouper = ds.groupby('time.dayofyear')
        elif aggregation == 'weekly':
            grouper = ds.resample(time='1W')
        elif aggregation == 'monthly':
            grouper = ds.resample(time='1M')
        elif aggregation == 'seasonal':
            grouper = ds.groupby('time.season')
        elif aggregation == 'annual':
            grouper = ds.groupby('time.year')
        else:
            raise ValueError(f"Unknown aggregation period: {aggregation}")
        
        # Calculate statistics
        aggregated_data = {}
        
        for var_name in ['methane_mixing_ratio', 'methane_anomaly']:
            if var_name not in ds.variables:
                continue
                
            for stat in statistics:
                if stat == 'mean':
                    result = grouper.mean()
                elif stat == 'max':
                    result = grouper.max()
                elif stat == 'min':
                    result = grouper.min()
                elif stat == 'std':
                    result = grouper.std()
                elif stat == 'median':
                    result = grouper.median()
                else:
                    continue
                
                new_var_name = f"{var_name}_{stat}"
                if stat in ['mean', 'median'] and aggregation == 'monthly':
                    new_var_name = f"{var_name}_monthly_{stat}"
                
                aggregated_data[new_var_name] = result[var_name]
        
        # Create new dataset
        aggregated_ds = xr.Dataset(aggregated_data)
        
        # Copy attributes
        aggregated_ds.attrs = ds.attrs.copy()
        aggregated_ds.attrs['temporal_aggregation'] = aggregation
        aggregated_ds.attrs['statistics'] = statistics
        
        self.logger.info(f"Completed {aggregation} aggregation")
        return aggregated_ds
    
    @log_function_call
    def spatial_interpolation(self, ds: xr.Dataset, 
                            target_resolution: float = 0.01,
                            method: str = 'linear') -> xr.Dataset:
        """
        Spatially interpolate methane data to higher resolution.
        
        Args:
            ds: Input dataset
            target_resolution: Target spatial resolution in degrees
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            Interpolated dataset
        """
        self.logger.info(f"Performing spatial interpolation to {target_resolution}° resolution")
        
        # Get current coordinate ranges
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
        
        # Create new coordinate arrays
        new_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
        new_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
        
        # Interpolate to new grid
        interpolated_ds = ds.interp(
            latitude=new_lats,
            longitude=new_lons,
            method=method
        )
        
        # Update attributes
        interpolated_ds.attrs = ds.attrs.copy()
        interpolated_ds.attrs['spatial_resolution'] = f"{target_resolution} degrees"
        interpolated_ds.attrs['interpolation_method'] = method
        
        self.logger.info(f"Completed spatial interpolation to {len(new_lats)}x{len(new_lons)} grid")
        return interpolated_ds
    
    @log_function_call
    def export_to_raster(self, ds: xr.Dataset, variable: str, 
                        output_path: Union[str, Path],
                        time_slice: Optional[int] = None,
                        crs: str = 'EPSG:4326') -> Path:
        """
        Export dataset variable to raster format.
        
        Args:
            ds: Input dataset
            variable: Variable name to export
            output_path: Output file path
            time_slice: Time index to export (if None, exports mean over time)
            crs: Coordinate reference system
            
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting {variable} to {output_path}")
        
        if variable not in ds.variables:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Select data to export
        if time_slice is not None:
            data_array = ds[variable].isel(time=time_slice)
        else:
            # Calculate temporal mean
            data_array = ds[variable].mean(dim='time')
        
        # Convert to numpy array and handle coordinates
        data = data_array.values
        lats = data_array.latitude.values
        lons = data_array.longitude.values
        
        # Create affine transform
        lat_res = abs(lats[1] - lats[0])
        lon_res = abs(lons[1] - lons[0])
        
        transform = rasterio.transform.from_bounds(
            lons.min() - lon_res/2, lats.min() - lat_res/2,
            lons.max() + lon_res/2, lats.max() + lat_res/2,
            data.shape[1], data.shape[0]
        )
        
        # Write to raster
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(data, 1)
            
            # Add metadata
            dst.update_tags(
                variable_name=variable,
                source='TROPOMI methane data',
                units=data_array.attrs.get('units', 'ppb')
            )
        
        self.logger.info(f"Exported raster to {output_path}")
        return output_path
    
    @log_function_call
    def process_satellite_data(self, data_sources: List, 
                             output_dir: Union[str, Path],
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all satellite methane data and generate analysis products.
        
        Args:
            data_sources: List of data sources (files, directories, or configs)
            output_dir: Directory for output files
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with processed datasets and file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Processing satellite methane data")
        
        # Load and combine all data sources
        all_datasets = []
        for source in data_sources:
            try:
                ds = self.load_tropomi_data(source, start_date, end_date)
                all_datasets.append(ds)
            except Exception as e:
                self.logger.error(f"Failed to load data source {source}: {e}")
                continue
        
        if not all_datasets:
            # Create sample data if no sources available
            sample_config = self.config.get('study_area', {})
            ds = self.load_tropomi_data(sample_config, start_date, end_date)
        else:
            # Combine datasets
            ds = xr.concat(all_datasets, dim='time')
        
        # Calculate anomalies
        ds_with_anomalies = self.calculate_methane_anomalies(ds)
        
        # Temporal aggregation
        monthly_ds = self.temporal_aggregation(ds_with_anomalies, 'monthly')
        seasonal_ds = self.temporal_aggregation(ds_with_anomalies, 'seasonal')
        
        # Export rasters
        raster_exports = {}
        
        # Mean methane concentration
        mean_ch4_path = self.export_to_raster(
            ds_with_anomalies, 'methane_mixing_ratio', 
            output_dir / 'mean_methane_concentration.tif'
        )
        raster_exports['mean_concentration'] = mean_ch4_path
        
        # Mean anomaly
        if 'methane_anomaly' in ds_with_anomalies.variables:
            anomaly_path = self.export_to_raster(
                ds_with_anomalies, 'methane_anomaly',
                output_dir / 'mean_methane_anomaly.tif'
            )
            raster_exports['mean_anomaly'] = anomaly_path
        
        # Maximum monthly concentration
        if 'methane_mixing_ratio_max' in monthly_ds.variables:
            max_monthly_path = self.export_to_raster(
                monthly_ds, 'methane_mixing_ratio_max',
                output_dir / 'max_monthly_methane.tif'
            )
            raster_exports['max_monthly'] = max_monthly_path
        
        # Save processed datasets
        processed_data_path = output_dir / 'processed_methane_data.nc'
        ds_with_anomalies.to_netcdf(processed_data_path)
        
        monthly_data_path = output_dir / 'monthly_methane_data.nc'
        monthly_ds.to_netcdf(monthly_data_path)
        
        results = {
            'full_dataset': ds_with_anomalies,
            'monthly_dataset': monthly_ds,
            'seasonal_dataset': seasonal_ds,
            'processed_data_path': processed_data_path,
            'monthly_data_path': monthly_data_path,
            'raster_exports': raster_exports,
            'temporal_extent': {
                'start': str(ds_with_anomalies.time.min().values),
                'end': str(ds_with_anomalies.time.max().values)
            },
            'spatial_extent': {
                'lat_min': float(ds_with_anomalies.latitude.min()),
                'lat_max': float(ds_with_anomalies.latitude.max()),
                'lon_min': float(ds_with_anomalies.longitude.min()),
                'lon_max': float(ds_with_anomalies.longitude.max())
            },
            'total_observations': int(ds_with_anomalies.methane_mixing_ratio.count())
        }
        
        self.logger.info("Completed processing satellite methane data")
        return results