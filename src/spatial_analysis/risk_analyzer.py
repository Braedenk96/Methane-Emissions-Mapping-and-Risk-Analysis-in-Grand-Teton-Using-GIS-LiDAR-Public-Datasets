"""
Risk analysis module for methane emissions assessment.

This module implements the core risk modeling and assessment algorithms
that combine multiple datasets to evaluate methane emissions risk.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any
from shapely.geometry import Point, Polygon
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings

from ..utils import Logger, log_function_call, ProgressLogger, DataValidator


class RiskAnalyzer:
    """Core risk analysis engine for methane emissions assessment."""
    
    def __init__(self, config: Dict):
        """
        Initialize risk analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.validator = DataValidator()
        
        # Risk factor weights from configuration
        self.risk_weights = config.get('analysis', {}).get('risk_factors', {}).get('weights', {
            'infrastructure_proximity': 0.35,
            'methane_concentration': 0.30,
            'terrain_factors': 0.15,
            'meteorological': 0.10,
            'landcover_sensitivity': 0.10
        })
        
        # Risk classification thresholds
        self.risk_thresholds = config.get('analysis', {}).get('risk_classification', {}).get('thresholds', {
            'very_low': 0.2,
            'low': 0.4,
            'moderate': 0.6,
            'high': 0.8,
            'very_high': 1.0
        })
        
        # Initialize scalers for data normalization
        self.scaler = MinMaxScaler()
        
    @log_function_call
    def calculate_infrastructure_risk(self, point_locations: gpd.GeoDataFrame,
                                    wells_gdf: gpd.GeoDataFrame,
                                    pipelines_gdf: gpd.GeoDataFrame,
                                    buffer_zones: Dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
        """
        Calculate infrastructure proximity risk scores.
        
        Args:
            point_locations: Analysis points (grid cells or specific locations)
            wells_gdf: Oil & gas wells data
            pipelines_gdf: Pipeline infrastructure data
            buffer_zones: Infrastructure buffer zones
            
        Returns:
            DataFrame with infrastructure risk scores
        """
        self.logger.info("Calculating infrastructure proximity risk")
        
        # Convert to UTM for accurate distance calculations
        points_utm = point_locations.to_crs('EPSG:32612')
        wells_utm = wells_gdf.to_crs('EPSG:32612') if len(wells_gdf) > 0 else gpd.GeoDataFrame()
        pipelines_utm = pipelines_gdf.to_crs('EPSG:32612') if len(pipelines_gdf) > 0 else gpd.GeoDataFrame()
        
        risk_scores = []
        
        progress = ProgressLogger(len(point_locations), "Infrastructure Risk Calculation")
        
        for idx, point in points_utm.iterrows():
            point_geom = point.geometry
            
            # Well proximity risk
            well_risk = 0.0
            if len(wells_utm) > 0:
                # Calculate distances to all wells
                well_distances = wells_utm.geometry.distance(point_geom)
                
                # Weight by well risk scores and inverse distance
                for well_idx, well_row in wells_utm.iterrows():
                    distance = well_distances[well_idx]
                    if distance > 0:
                        # Exponential decay with distance
                        distance_factor = np.exp(-distance / 1000)  # 1km decay constant
                        well_individual_risk = well_row.get('leak_risk_score', 0.5)
                        well_risk += well_individual_risk * distance_factor
                
                # Normalize well risk
                well_risk = min(well_risk, 1.0)
            
            # Pipeline proximity risk
            pipeline_risk = 0.0
            if len(pipelines_utm) > 0:
                # Calculate distances to all pipelines
                pipeline_distances = pipelines_utm.geometry.distance(point_geom)
                
                for pipe_idx, pipe_row in pipelines_utm.iterrows():
                    distance = pipeline_distances[pipe_idx]
                    if distance > 0:
                        # Linear decay with distance (pipelines have broader impact)
                        distance_factor = max(0, 1 - distance / 2000)  # 2km linear decay
                        pipe_individual_risk = pipe_row.get('leak_risk_score', 0.5)
                        pipeline_risk += pipe_individual_risk * distance_factor
                
                # Normalize pipeline risk
                pipeline_risk = min(pipeline_risk, 1.0)
            
            # Combined infrastructure risk
            infrastructure_risk = (well_risk * 0.6 + pipeline_risk * 0.4)  # Wells weighted higher
            
            # Buffer zone analysis
            buffer_risk_modifiers = {}
            for buffer_name, buffer_gdf in buffer_zones.items():
                if len(buffer_gdf) > 0:
                    # Check if point is within any buffer zone
                    buffer_utm = buffer_gdf.to_crs('EPSG:32612')
                    within_buffer = buffer_utm.geometry.contains(point_geom).any()
                    buffer_risk_modifiers[buffer_name] = within_buffer
            
            # Apply buffer zone modifiers
            if buffer_risk_modifiers.get('high_risk_combined', False):
                infrastructure_risk = min(infrastructure_risk * 1.5, 1.0)  # 50% increase
            
            risk_scores.append({
                'point_id': idx,
                'well_proximity_risk': well_risk,
                'pipeline_proximity_risk': pipeline_risk,
                'infrastructure_risk_score': infrastructure_risk,
                'within_high_risk_zone': buffer_risk_modifiers.get('high_risk_combined', False),
                'nearest_well_distance_m': well_distances.min() if len(wells_utm) > 0 else np.inf,
                'nearest_pipeline_distance_m': pipeline_distances.min() if len(pipelines_utm) > 0 else np.inf
            })
            
            if idx % 100 == 0:
                progress.update(100)
        
        progress.complete()
        
        risk_df = pd.DataFrame(risk_scores)
        self.logger.info(f"Calculated infrastructure risk for {len(risk_df)} points")
        
        return risk_df
    
    @log_function_call
    def calculate_methane_concentration_risk(self, point_locations: gpd.GeoDataFrame,
                                           methane_data: xr.Dataset) -> pd.DataFrame:
        """
        Calculate risk scores based on methane concentrations.
        
        Args:
            point_locations: Analysis points
            methane_data: Satellite methane concentration data
            
        Returns:
            DataFrame with methane concentration risk scores
        """
        self.logger.info("Calculating methane concentration risk")
        
        risk_scores = []
        
        # Extract methane variables
        if 'methane_mixing_ratio' in methane_data.variables:
            ch4_var = 'methane_mixing_ratio'
        else:
            available_vars = list(methane_data.variables)
            self.logger.warning(f"No methane_mixing_ratio found. Available variables: {available_vars}")
            # Return empty risk scores
            return pd.DataFrame([{
                'point_id': idx,
                'mean_methane_concentration': np.nan,
                'max_methane_concentration': np.nan,
                'methane_anomaly_mean': np.nan,
                'methane_concentration_risk': 0.0
            } for idx in range(len(point_locations))])
        
        progress = ProgressLogger(len(point_locations), "Methane Risk Calculation")
        
        for idx, point in point_locations.iterrows():
            point_geom = point.geometry
            lon, lat = point_geom.x, point_geom.y
            
            try:
                # Find nearest grid cell in methane data
                point_data = methane_data.sel(
                    longitude=lon, latitude=lat, 
                    method='nearest'
                )
                
                # Calculate temporal statistics
                ch4_values = point_data[ch4_var].values
                
                # Handle NaN values
                valid_ch4 = ch4_values[~np.isnan(ch4_values)]
                
                if len(valid_ch4) == 0:
                    # No valid methane data at this location
                    risk_scores.append({
                        'point_id': idx,
                        'mean_methane_concentration': np.nan,
                        'max_methane_concentration': np.nan,
                        'methane_anomaly_mean': np.nan,
                        'methane_concentration_risk': 0.0
                    })
                    continue
                
                mean_ch4 = np.mean(valid_ch4)
                max_ch4 = np.max(valid_ch4)
                std_ch4 = np.std(valid_ch4)
                
                # Calculate anomaly risk
                background_ch4 = 1850  # ppb
                mean_anomaly = mean_ch4 - background_ch4
                max_anomaly = max_ch4 - background_ch4
                
                # Risk based on concentration levels
                # Higher concentrations = higher risk
                concentration_risk = 0.0
                
                if mean_ch4 > background_ch4 + 50:  # 50 ppb above background
                    concentration_risk += 0.3
                
                if mean_ch4 > background_ch4 + 100:  # 100 ppb above background
                    concentration_risk += 0.3
                
                if max_ch4 > background_ch4 + 200:  # Very high peaks
                    concentration_risk += 0.4
                
                # Add variability factor (high variability suggests emission sources)
                if std_ch4 > 30:  # High temporal variability
                    concentration_risk += 0.2
                
                # Normalize to 0-1 range
                concentration_risk = min(concentration_risk, 1.0)
                
                # Anomaly analysis if available
                anomaly_mean = np.nan
                if 'methane_anomaly' in point_data.variables:
                    anomaly_values = point_data['methane_anomaly'].values
                    valid_anomalies = anomaly_values[~np.isnan(anomaly_values)]
                    if len(valid_anomalies) > 0:
                        anomaly_mean = np.mean(valid_anomalies)
                        
                        # Boost risk score for persistent anomalies
                        if anomaly_mean > 50:  # Persistent anomaly > 50 ppb
                            concentration_risk = min(concentration_risk * 1.3, 1.0)
                
                risk_scores.append({
                    'point_id': idx,
                    'mean_methane_concentration': mean_ch4,
                    'max_methane_concentration': max_ch4,
                    'std_methane_concentration': std_ch4,
                    'methane_anomaly_mean': anomaly_mean,
                    'methane_concentration_risk': concentration_risk
                })
                
            except Exception as e:
                self.logger.warning(f"Could not extract methane data for point {idx}: {e}")
                risk_scores.append({
                    'point_id': idx,
                    'mean_methane_concentration': np.nan,
                    'max_methane_concentration': np.nan,
                    'methane_anomaly_mean': np.nan,
                    'methane_concentration_risk': 0.0
                })
            
            if idx % 50 == 0:
                progress.update(50)
        
        progress.complete()
        
        risk_df = pd.DataFrame(risk_scores)
        self.logger.info(f"Calculated methane concentration risk for {len(risk_df)} points")
        
        return risk_df
    
    @log_function_call
    def calculate_terrain_risk(self, point_locations: gpd.GeoDataFrame,
                             dtm_path: Optional[Union[str, Path]] = None,
                             slope_threshold: float = 15.0) -> pd.DataFrame:
        """
        Calculate terrain-based risk factors.
        
        Args:
            point_locations: Analysis points
            dtm_path: Path to Digital Terrain Model raster
            slope_threshold: Slope threshold in degrees for high-risk areas
            
        Returns:
            DataFrame with terrain risk scores
        """
        self.logger.info("Calculating terrain-based risk factors")
        
        risk_scores = []
        
        if dtm_path and Path(dtm_path).exists():
            # Load DTM and calculate terrain metrics
            with rasterio.open(dtm_path) as dtm_src:
                dtm_data = dtm_src.read(1)
                dtm_transform = dtm_src.transform
                
                # Calculate slope and aspect
                pixel_size_x = dtm_transform.a  # pixel width
                pixel_size_y = -dtm_transform.e  # pixel height (positive)
                
                # Simple gradient calculation
                dy, dx = np.gradient(dtm_data, pixel_size_y, pixel_size_x)
                slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi  # Convert to degrees
                
                progress = ProgressLogger(len(point_locations), "Terrain Risk Calculation")
                
                for idx, point in point_locations.iterrows():
                    point_geom = point.geometry
                    
                    try:
                        # Get pixel coordinates
                        row, col = rasterio.transform.rowcol(
                            dtm_transform, point_geom.x, point_geom.y
                        )
                        
                        # Check bounds
                        if (0 <= row < dtm_data.shape[0] and 0 <= col < dtm_data.shape[1]):
                            elevation = dtm_data[row, col]
                            point_slope = slope[row, col]
                            
                            # Calculate terrain risk factors
                            # Higher elevations may have different wind patterns
                            elevation_factor = min(elevation / 3000, 1.0)  # Normalize by ~3000m
                            
                            # Steep slopes may affect gas dispersion
                            slope_factor = 0.0
                            if point_slope > slope_threshold:
                                slope_factor = min((point_slope - slope_threshold) / 30, 0.5)
                            
                            # Combined terrain risk (generally lower impact than other factors)
                            terrain_risk = (elevation_factor * 0.3 + slope_factor * 0.7) * 0.5
                            
                            risk_scores.append({
                                'point_id': idx,
                                'elevation_m': elevation,
                                'slope_degrees': point_slope,
                                'elevation_factor': elevation_factor,
                                'slope_factor': slope_factor,
                                'terrain_risk_score': terrain_risk
                            })
                        
                        else:
                            # Point outside DTM bounds
                            risk_scores.append({
                                'point_id': idx,
                                'elevation_m': np.nan,
                                'slope_degrees': np.nan,
                                'elevation_factor': 0.0,
                                'slope_factor': 0.0,
                                'terrain_risk_score': 0.0
                            })
                    
                    except Exception as e:
                        self.logger.warning(f"Could not extract terrain data for point {idx}: {e}")
                        risk_scores.append({
                            'point_id': idx,
                            'elevation_m': np.nan,
                            'slope_degrees': np.nan,
                            'elevation_factor': 0.0,
                            'slope_factor': 0.0,
                            'terrain_risk_score': 0.0
                        })
                    
                    if idx % 100 == 0:
                        progress.update(100)
                
                progress.complete()
        
        else:
            # No DTM available - use default low risk
            self.logger.warning("No DTM data available, using default terrain risk scores")
            for idx in range(len(point_locations)):
                risk_scores.append({
                    'point_id': idx,
                    'elevation_m': np.nan,
                    'slope_degrees': np.nan,
                    'elevation_factor': 0.0,
                    'slope_factor': 0.0,
                    'terrain_risk_score': 0.1  # Low default risk
                })
        
        risk_df = pd.DataFrame(risk_scores)
        self.logger.info(f"Calculated terrain risk for {len(risk_df)} points")
        
        return risk_df
    
    @log_function_call
    def calculate_landcover_sensitivity(self, point_locations: gpd.GeoDataFrame,
                                      landcover_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Calculate environmental sensitivity based on land cover.
        
        Args:
            point_locations: Analysis points
            landcover_path: Path to land cover raster
            
        Returns:
            DataFrame with land cover sensitivity scores
        """
        self.logger.info("Calculating land cover sensitivity")
        
        # Land cover sensitivity weights
        # Higher values indicate more sensitive environments
        landcover_sensitivity = {
            11: 0.9,   # Open Water - high sensitivity
            12: 0.8,   # Perennial Ice/Snow - high sensitivity
            21: 0.1,   # Developed, Open Space - low sensitivity
            22: 0.05,  # Developed, Low Intensity - very low sensitivity
            23: 0.02,  # Developed, Medium Intensity - very low sensitivity
            24: 0.01,  # Developed, High Intensity - very low sensitivity
            31: 0.3,   # Barren Land - moderate sensitivity
            41: 0.7,   # Deciduous Forest - high sensitivity
            42: 0.7,   # Evergreen Forest - high sensitivity
            43: 0.7,   # Mixed Forest - high sensitivity
            51: 0.6,   # Dwarf Scrub - moderate-high sensitivity
            52: 0.5,   # Shrub/Scrub - moderate sensitivity
            71: 0.4,   # Grassland/Herbaceous - moderate sensitivity
            72: 0.3,   # Sedge/Herbaceous - moderate sensitivity
            73: 0.3,   # Lichens - moderate sensitivity
            74: 0.3,   # Moss - moderate sensitivity
            81: 0.2,   # Pasture/Hay - low-moderate sensitivity
            82: 0.2,   # Cultivated Crops - low-moderate sensitivity
            90: 0.9,   # Woody Wetlands - very high sensitivity
            95: 0.9    # Emergent Herbaceous Wetlands - very high sensitivity
        }
        
        risk_scores = []
        
        if landcover_path and Path(landcover_path).exists():
            with rasterio.open(landcover_path) as lc_src:
                lc_data = lc_src.read(1)
                lc_transform = lc_src.transform
                
                progress = ProgressLogger(len(point_locations), "Land Cover Sensitivity")
                
                for idx, point in point_locations.iterrows():
                    point_geom = point.geometry
                    
                    try:
                        # Get pixel coordinates
                        row, col = rasterio.transform.rowcol(
                            lc_transform, point_geom.x, point_geom.y
                        )
                        
                        # Check bounds
                        if (0 <= row < lc_data.shape[0] and 0 <= col < lc_data.shape[1]):
                            lc_class = lc_data[row, col]
                            sensitivity = landcover_sensitivity.get(lc_class, 0.3)  # Default moderate sensitivity
                            
                            risk_scores.append({
                                'point_id': idx,
                                'landcover_class': lc_class,
                                'sensitivity_score': sensitivity
                            })
                        
                        else:
                            # Point outside raster bounds
                            risk_scores.append({
                                'point_id': idx,
                                'landcover_class': np.nan,
                                'sensitivity_score': 0.3  # Default
                            })
                    
                    except Exception as e:
                        self.logger.warning(f"Could not extract land cover for point {idx}: {e}")
                        risk_scores.append({
                            'point_id': idx,
                            'landcover_class': np.nan,
                            'sensitivity_score': 0.3  # Default
                        })
                    
                    if idx % 100 == 0:
                        progress.update(100)
                
                progress.complete()
        
        else:
            # No land cover data - use default sensitivity
            self.logger.warning("No land cover data available, using default sensitivity scores")
            for idx in range(len(point_locations)):
                risk_scores.append({
                    'point_id': idx,
                    'landcover_class': np.nan,
                    'sensitivity_score': 0.3  # Default moderate sensitivity
                })
        
        risk_df = pd.DataFrame(risk_scores)
        self.logger.info(f"Calculated land cover sensitivity for {len(risk_df)} points")
        
        return risk_df
    
    @log_function_call
    def integrate_risk_factors(self, infrastructure_risk: pd.DataFrame,
                             methane_risk: pd.DataFrame,
                             terrain_risk: pd.DataFrame,
                             landcover_risk: pd.DataFrame,
                             meteorological_risk: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Integrate all risk factors into final risk assessment.
        
        Args:
            infrastructure_risk: Infrastructure proximity risk scores
            methane_risk: Methane concentration risk scores
            terrain_risk: Terrain-based risk scores
            landcover_risk: Land cover sensitivity scores
            meteorological_risk: Weather-based risk scores (optional)
            
        Returns:
            DataFrame with integrated risk assessment
        """
        self.logger.info("Integrating risk factors")
        
        # Merge all risk dataframes
        integrated = infrastructure_risk[['point_id', 'infrastructure_risk_score']].copy()
        
        # Merge methane risk
        integrated = integrated.merge(
            methane_risk[['point_id', 'methane_concentration_risk']], 
            on='point_id', how='left'
        )
        
        # Merge terrain risk
        integrated = integrated.merge(
            terrain_risk[['point_id', 'terrain_risk_score']], 
            on='point_id', how='left'
        )
        
        # Merge land cover sensitivity
        integrated = integrated.merge(
            landcover_risk[['point_id', 'sensitivity_score']], 
            on='point_id', how='left'
        )
        
        # Fill NaN values with 0
        integrated = integrated.fillna(0)
        
        # Add meteorological risk if available
        meteorological_score = 0.1  # Default low meteorological risk
        if meteorological_risk is not None:
            integrated = integrated.merge(
                meteorological_risk[['point_id', 'meteorological_risk']], 
                on='point_id', how='left'
            )
            integrated['meteorological_risk'] = integrated['meteorological_risk'].fillna(meteorological_score)
        else:
            integrated['meteorological_risk'] = meteorological_score
        
        # Calculate weighted composite risk score
        integrated['composite_risk_score'] = (
            integrated['infrastructure_risk_score'] * self.risk_weights.get('infrastructure_proximity', 0.35) +
            integrated['methane_concentration_risk'] * self.risk_weights.get('methane_concentration', 0.30) +
            integrated['terrain_risk_score'] * self.risk_weights.get('terrain_factors', 0.15) +
            integrated['meteorological_risk'] * self.risk_weights.get('meteorological', 0.10) +
            integrated['sensitivity_score'] * self.risk_weights.get('landcover_sensitivity', 0.10)
        )
        
        # Classify risk levels
        integrated['risk_category'] = integrated['composite_risk_score'].apply(self._classify_risk_level)
        
        # Calculate confidence score based on data availability
        data_availability_score = (
            (~integrated['infrastructure_risk_score'].isna()).astype(int) * 0.35 +
            (~integrated['methane_concentration_risk'].isna()).astype(int) * 0.30 +
            (~integrated['terrain_risk_score'].isna()).astype(int) * 0.15 +
            (~integrated['meteorological_risk'].isna()).astype(int) * 0.10 +
            (~integrated['sensitivity_score'].isna()).astype(int) * 0.10
        )
        
        integrated['confidence_score'] = data_availability_score
        
        # Add priority ranking
        integrated['priority_rank'] = integrated['composite_risk_score'].rank(method='dense', ascending=False)
        
        self.logger.info(f"Integrated risk assessment for {len(integrated)} points")
        
        # Log risk category distribution
        risk_counts = integrated['risk_category'].value_counts()
        self.logger.info(f"Risk category distribution: {risk_counts.to_dict()}")
        
        return integrated
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk score into categorical risk level.
        
        Args:
            risk_score: Composite risk score (0-1)
            
        Returns:
            Risk category string
        """
        if risk_score <= self.risk_thresholds['very_low']:
            return 'very_low'
        elif risk_score <= self.risk_thresholds['low']:
            return 'low'
        elif risk_score <= self.risk_thresholds['moderate']:
            return 'moderate'
        elif risk_score <= self.risk_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    @log_function_call
    def generate_risk_statistics(self, integrated_risk: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for risk assessment.
        
        Args:
            integrated_risk: Integrated risk assessment DataFrame
            
        Returns:
            Dictionary with risk statistics
        """
        self.logger.info("Generating risk assessment statistics")
        
        stats = {}
        
        # Overall statistics
        stats['total_points'] = len(integrated_risk)
        stats['mean_risk_score'] = integrated_risk['composite_risk_score'].mean()
        stats['std_risk_score'] = integrated_risk['composite_risk_score'].std()
        stats['max_risk_score'] = integrated_risk['composite_risk_score'].max()
        stats['min_risk_score'] = integrated_risk['composite_risk_score'].min()
        
        # Risk category distribution
        risk_distribution = integrated_risk['risk_category'].value_counts()
        stats['risk_distribution'] = risk_distribution.to_dict()
        stats['risk_distribution_percent'] = (risk_distribution / len(integrated_risk) * 100).to_dict()
        
        # High priority areas (top 10%)
        high_priority_threshold = integrated_risk['composite_risk_score'].quantile(0.9)
        high_priority_points = integrated_risk[integrated_risk['composite_risk_score'] >= high_priority_threshold]
        stats['high_priority_points'] = len(high_priority_points)
        stats['high_priority_threshold'] = high_priority_threshold
        
        # Data quality assessment
        stats['mean_confidence_score'] = integrated_risk['confidence_score'].mean()
        stats['low_confidence_points'] = len(integrated_risk[integrated_risk['confidence_score'] < 0.7])
        
        # Risk factor contributions
        factor_correlations = {}
        for factor in ['infrastructure_risk_score', 'methane_concentration_risk', 
                      'terrain_risk_score', 'meteorological_risk', 'sensitivity_score']:
            if factor in integrated_risk.columns:
                correlation = integrated_risk[factor].corr(integrated_risk['composite_risk_score'])
                factor_correlations[factor] = correlation
        
        stats['factor_correlations'] = factor_correlations
        
        # Spatial statistics (if coordinates available)
        if 'geometry' in integrated_risk.columns or ('x' in integrated_risk.columns and 'y' in integrated_risk.columns):
            high_risk_areas = integrated_risk[integrated_risk['risk_category'].isin(['high', 'very_high'])]
            stats['high_risk_area_count'] = len(high_risk_areas)
            
            if len(high_risk_areas) > 0:
                # Calculate spatial clustering metrics could be added here
                stats['high_risk_area_percentage'] = len(high_risk_areas) / len(integrated_risk) * 100
        
        self.logger.info(f"Generated comprehensive risk statistics")
        return stats