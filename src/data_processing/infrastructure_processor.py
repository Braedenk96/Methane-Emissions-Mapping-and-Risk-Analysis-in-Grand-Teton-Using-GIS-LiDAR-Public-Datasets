"""
Oil & Gas infrastructure data processing module for methane emissions analysis.

This module handles the processing of oil & gas infrastructure datasets
including wells, pipelines, and facilities for methane emissions risk assessment.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import requests
from shapely.geometry import Point, Polygon, LineString, buffer
from shapely.ops import unary_union
import json

from ..utils import Logger, log_function_call, ProgressLogger, DataValidator


class InfrastructureProcessor:
    """Processes oil & gas infrastructure data for methane emissions analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize infrastructure processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.validator = DataValidator()
        
        # Processing parameters
        self.well_buffer_distances = config.get('proximity_analysis', {}).get('well_buffer_m', [100, 500, 1000, 2000])
        self.pipeline_buffer_distances = config.get('proximity_analysis', {}).get('pipeline_buffer_m', [50, 200, 500, 1000])
        
    @log_function_call
    def load_well_data(self, data_source: Union[str, Path, Dict]) -> gpd.GeoDataFrame:
        """
        Load oil & gas well data from various sources.
        
        Args:
            data_source: Path to data file, API endpoint, or data dict
            
        Returns:
            GeoDataFrame with well locations and attributes
        """
        self.logger.info("Loading oil & gas well data")
        
        if isinstance(data_source, (str, Path)):
            # Load from file
            data_path = Path(data_source)
            
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
                # Assume columns 'longitude', 'latitude' exist
                if 'longitude' in df.columns and 'latitude' in df.columns:
                    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
                else:
                    raise ValueError("CSV file must contain 'longitude' and 'latitude' columns")
            
            elif data_path.suffix.lower() in ['.shp', '.geojson', '.gpkg']:
                gdf = gpd.read_file(data_path)
            
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        elif isinstance(data_source, dict):
            # Create sample data for demonstration
            gdf = self._create_sample_well_data(data_source)
        
        else:
            raise ValueError("data_source must be file path or dictionary")
        
        # Validate and standardize well data
        gdf = self._standardize_well_data(gdf)
        
        self.logger.info(f"Loaded {len(gdf)} wells")
        return gdf
    
    def _create_sample_well_data(self, bounds_config: Dict) -> gpd.GeoDataFrame:
        """
        Create sample well data for demonstration purposes.
        
        Args:
            bounds_config: Dictionary with bounding box information
            
        Returns:
            GeoDataFrame with sample well data
        """
        bbox = bounds_config.get('bbox', [-111.05, 43.65, -110.10, 43.85])
        minx, miny, maxx, maxy = bbox
        
        # Generate random well locations within study area
        np.random.seed(42)  # For reproducible results
        n_wells = 50
        
        well_data = []
        well_types = ['Oil', 'Gas', 'Oil & Gas', 'Injection', 'Dry Hole']
        well_statuses = ['Active', 'Inactive', 'Plugged', 'Drilling']
        
        for i in range(n_wells):
            lon = np.random.uniform(minx, maxx)
            lat = np.random.uniform(miny, maxy)
            
            well_data.append({
                'well_id': f"WELL_{i+1:03d}",
                'api_number': f"49{np.random.randint(1000000, 9999999)}",
                'well_name': f"Grand Teton Well #{i+1}",
                'longitude': lon,
                'latitude': lat,
                'well_type': np.random.choice(well_types),
                'status': np.random.choice(well_statuses),
                'spud_date': pd.date_range('2000-01-01', '2023-12-31', periods=1)[0],
                'total_depth': np.random.uniform(1000, 15000),
                'surface_elevation': np.random.uniform(6000, 8000),
                'operator': f"Operator_{np.random.randint(1, 10)}",
                'daily_production_oil': np.random.uniform(0, 500) if 'Oil' in np.random.choice(well_types) else 0,
                'daily_production_gas': np.random.uniform(0, 10000),
                'leak_risk_score': np.random.uniform(0.1, 0.9)
            })
        
        df = pd.DataFrame(well_data)
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def _standardize_well_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Standardize well data columns and values.
        
        Args:
            gdf: Raw well GeoDataFrame
            
        Returns:
            Standardized GeoDataFrame
        """
        # Ensure required columns exist
        required_columns = ['well_id', 'well_type', 'status']
        for col in required_columns:
            if col not in gdf.columns:
                if col == 'well_id':
                    gdf['well_id'] = [f"WELL_{i}" for i in range(len(gdf))]
                elif col == 'well_type':
                    gdf['well_type'] = 'Unknown'
                elif col == 'status':
                    gdf['status'] = 'Unknown'
        
        # Standardize status values
        status_mapping = {
            'active': 'Active',
            'producing': 'Active',
            'inactive': 'Inactive',
            'shut-in': 'Inactive',
            'plugged': 'Plugged',
            'abandoned': 'Plugged',
            'drilling': 'Drilling'
        }
        
        gdf['status'] = gdf['status'].astype(str).str.lower().map(
            lambda x: status_mapping.get(x, x.title())
        )
        
        # Add risk indicators
        if 'leak_risk_score' not in gdf.columns:
            # Calculate simple risk score based on age and status
            current_year = pd.Timestamp.now().year
            if 'spud_date' in gdf.columns:
                gdf['well_age'] = current_year - pd.to_datetime(gdf['spud_date']).dt.year
                # Higher risk for older wells and certain statuses
                base_risk = np.random.uniform(0.1, 0.3, len(gdf))
                age_factor = np.clip(gdf['well_age'] / 30, 0, 0.4)  # Max 0.4 for very old wells
                status_factor = gdf['status'].map({
                    'Active': 0.1,
                    'Inactive': 0.3,
                    'Plugged': 0.05,
                    'Drilling': 0.2
                }).fillna(0.2)
                
                gdf['leak_risk_score'] = np.clip(base_risk + age_factor + status_factor, 0, 1)
            else:
                gdf['leak_risk_score'] = np.random.uniform(0.1, 0.8, len(gdf))
        
        return gdf
    
    @log_function_call
    def load_pipeline_data(self, data_source: Union[str, Path, Dict]) -> gpd.GeoDataFrame:
        """
        Load pipeline infrastructure data.
        
        Args:
            data_source: Path to data file or configuration dict
            
        Returns:
            GeoDataFrame with pipeline data
        """
        self.logger.info("Loading pipeline infrastructure data")
        
        if isinstance(data_source, (str, Path)):
            # Load from file
            data_path = Path(data_source)
            gdf = gpd.read_file(data_path)
        
        elif isinstance(data_source, dict):
            # Create sample data for demonstration
            gdf = self._create_sample_pipeline_data(data_source)
        
        else:
            raise ValueError("data_source must be file path or dictionary")
        
        # Validate and standardize pipeline data
        gdf = self._standardize_pipeline_data(gdf)
        
        self.logger.info(f"Loaded {len(gdf)} pipeline segments")
        return gdf
    
    def _create_sample_pipeline_data(self, bounds_config: Dict) -> gpd.GeoDataFrame:
        """
        Create sample pipeline data for demonstration.
        
        Args:
            bounds_config: Dictionary with bounding box information
            
        Returns:
            GeoDataFrame with sample pipeline data
        """
        bbox = bounds_config.get('bbox', [-111.05, 43.65, -110.10, 43.85])
        minx, miny, maxx, maxy = bbox
        
        np.random.seed(42)
        n_pipelines = 15
        
        pipeline_data = []
        pipeline_types = ['Natural Gas', 'Oil', 'Refined Products', 'CO2']
        operators = ['Pipeline Co A', 'Pipeline Co B', 'Pipeline Co C']
        
        for i in range(n_pipelines):
            # Create random pipeline segments
            n_points = np.random.randint(3, 8)
            
            start_lon = np.random.uniform(minx, maxx)
            start_lat = np.random.uniform(miny, maxy)
            
            coords = [(start_lon, start_lat)]
            
            for j in range(n_points - 1):
                # Add some variation to create realistic pipeline paths
                lon_delta = np.random.uniform(-0.05, 0.05)
                lat_delta = np.random.uniform(-0.05, 0.05)
                
                next_lon = coords[-1][0] + lon_delta
                next_lat = coords[-1][1] + lat_delta
                
                # Keep within bounds
                next_lon = np.clip(next_lon, minx, maxx)
                next_lat = np.clip(next_lat, miny, maxy)
                
                coords.append((next_lon, next_lat))
            
            geometry = LineString(coords)
            
            pipeline_data.append({
                'pipeline_id': f"PIPE_{i+1:03d}",
                'pipeline_name': f"Pipeline Segment {i+1}",
                'operator': np.random.choice(operators),
                'pipeline_type': np.random.choice(pipeline_types),
                'diameter_inches': np.random.choice([6, 8, 12, 16, 20, 24, 30, 36]),
                'install_year': np.random.randint(1950, 2020),
                'pressure_psi': np.random.uniform(200, 1500),
                'material': np.random.choice(['Steel', 'Plastic', 'Cast Iron']),
                'length_miles': geometry.length * 69,  # Rough conversion to miles
                'leak_risk_score': np.random.uniform(0.1, 0.8)
            })
        
        gdf = gpd.GeoDataFrame(pipeline_data, geometry=[p['geometry'] for p in pipeline_data], crs='EPSG:4326')
        gdf = gdf.drop('geometry', axis=1)  # Remove the dict entry
        gdf['geometry'] = [LineString(coords) for coords in [
            [(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)) for _ in range(np.random.randint(3, 8))]
            for _ in range(len(pipeline_data))
        ]]
        
        return gdf
    
    def _standardize_pipeline_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Standardize pipeline data columns and values.
        
        Args:
            gdf: Raw pipeline GeoDataFrame
            
        Returns:
            Standardized GeoDataFrame
        """
        # Ensure required columns exist
        if 'pipeline_id' not in gdf.columns:
            gdf['pipeline_id'] = [f"PIPE_{i}" for i in range(len(gdf))]
        
        if 'pipeline_type' not in gdf.columns:
            gdf['pipeline_type'] = 'Unknown'
        
        # Calculate pipeline length if not present
        if 'length_miles' not in gdf.columns:
            # Convert to UTM for accurate length calculation
            gdf_utm = gdf.to_crs('EPSG:32612')  # UTM Zone 12N for Wyoming
            gdf['length_miles'] = gdf_utm.geometry.length / 1609.34  # meters to miles
        
        # Add risk score if not present
        if 'leak_risk_score' not in gdf.columns:
            # Simple risk based on age and type
            current_year = pd.Timestamp.now().year
            
            if 'install_year' in gdf.columns:
                age = current_year - gdf['install_year']
                age_factor = np.clip(age / 50, 0, 0.5)  # Max 0.5 for very old pipelines
            else:
                age_factor = 0.3
            
            type_risk = gdf['pipeline_type'].map({
                'Natural Gas': 0.3,
                'Oil': 0.4,
                'Refined Products': 0.2,
                'CO2': 0.1
            }).fillna(0.25)
            
            base_risk = 0.2
            gdf['leak_risk_score'] = np.clip(base_risk + age_factor + type_risk, 0, 1)
        
        return gdf
    
    @log_function_call
    def create_infrastructure_buffers(self, wells_gdf: gpd.GeoDataFrame, 
                                    pipelines_gdf: gpd.GeoDataFrame) -> Dict[str, gpd.GeoDataFrame]:
        """
        Create buffer zones around infrastructure for proximity analysis.
        
        Args:
            wells_gdf: Wells GeoDataFrame
            pipelines_gdf: Pipelines GeoDataFrame
            
        Returns:
            Dictionary of buffer GeoDataFrames by distance
        """
        self.logger.info("Creating infrastructure buffer zones")
        
        # Convert to UTM for accurate distance calculations
        wells_utm = wells_gdf.to_crs('EPSG:32612')
        pipelines_utm = pipelines_gdf.to_crs('EPSG:32612')
        
        buffers = {}
        
        # Well buffers
        for distance in self.well_buffer_distances:
            self.logger.info(f"Creating {distance}m buffers around wells")
            
            well_buffers = wells_utm.copy()
            well_buffers['geometry'] = wells_utm.geometry.buffer(distance)
            well_buffers['buffer_distance_m'] = distance
            well_buffers['infrastructure_type'] = 'well'
            
            # Convert back to WGS84
            well_buffers = well_buffers.to_crs('EPSG:4326')
            buffers[f'wells_{distance}m'] = well_buffers
        
        # Pipeline buffers
        for distance in self.pipeline_buffer_distances:
            self.logger.info(f"Creating {distance}m buffers around pipelines")
            
            pipeline_buffers = pipelines_utm.copy()
            pipeline_buffers['geometry'] = pipelines_utm.geometry.buffer(distance)
            pipeline_buffers['buffer_distance_m'] = distance
            pipeline_buffers['infrastructure_type'] = 'pipeline'
            
            # Convert back to WGS84
            pipeline_buffers = pipeline_buffers.to_crs('EPSG:4326')
            buffers[f'pipelines_{distance}m'] = pipeline_buffers
        
        # Combined high-risk areas (union of all high-risk infrastructure)
        high_risk_wells = wells_utm[wells_utm['leak_risk_score'] > 0.7].copy()
        high_risk_pipelines = pipelines_utm[pipelines_utm['leak_risk_score'] > 0.7].copy()
        
        if len(high_risk_wells) > 0 or len(high_risk_pipelines) > 0:
            combined_geoms = []
            
            if len(high_risk_wells) > 0:
                well_union = unary_union(high_risk_wells.geometry.buffer(500))
                combined_geoms.append(well_union)
            
            if len(high_risk_pipelines) > 0:
                pipeline_union = unary_union(high_risk_pipelines.geometry.buffer(200))
                combined_geoms.append(pipeline_union)
            
            if combined_geoms:
                high_risk_union = unary_union(combined_geoms)
                
                high_risk_gdf = gpd.GeoDataFrame(
                    [{'risk_category': 'high', 'infrastructure_type': 'combined'}],
                    geometry=[high_risk_union],
                    crs='EPSG:32612'
                ).to_crs('EPSG:4326')
                
                buffers['high_risk_combined'] = high_risk_gdf
        
        self.logger.info(f"Created {len(buffers)} buffer zone datasets")
        return buffers
    
    @log_function_call
    def calculate_infrastructure_density(self, wells_gdf: gpd.GeoDataFrame,
                                       pipelines_gdf: gpd.GeoDataFrame,
                                       grid_size_km: float = 1.0,
                                       study_bounds: Optional[Tuple] = None) -> gpd.GeoDataFrame:
        """
        Calculate infrastructure density in grid cells.
        
        Args:
            wells_gdf: Wells GeoDataFrame
            pipelines_gdf: Pipelines GeoDataFrame
            grid_size_km: Grid cell size in kilometers
            study_bounds: Study area bounds [minx, miny, maxx, maxy]
            
        Returns:
            GeoDataFrame with infrastructure density metrics
        """
        self.logger.info(f"Calculating infrastructure density with {grid_size_km}km grid")
        
        if study_bounds is None:
            # Combine bounds from all infrastructure
            all_bounds = []
            if len(wells_gdf) > 0:
                all_bounds.append(wells_gdf.total_bounds)
            if len(pipelines_gdf) > 0:
                all_bounds.append(pipelines_gdf.total_bounds)
            
            if not all_bounds:
                raise ValueError("No infrastructure data available")
            
            combined_bounds = np.array(all_bounds)
            study_bounds = (
                combined_bounds[:, 0].min(),  # minx
                combined_bounds[:, 1].min(),  # miny
                combined_bounds[:, 2].max(),  # maxx
                combined_bounds[:, 3].max()   # maxy
            )
        
        minx, miny, maxx, maxy = study_bounds
        
        # Create grid
        grid_size_deg = grid_size_km / 111  # Rough conversion
        x_coords = np.arange(minx, maxx + grid_size_deg, grid_size_deg)
        y_coords = np.arange(miny, maxy + grid_size_deg, grid_size_deg)
        
        grid_cells = []
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                cell_bounds = [
                    x, y,
                    x_coords[i + 1], y_coords[j + 1]
                ]
                
                cell_polygon = Polygon([
                    (cell_bounds[0], cell_bounds[1]),
                    (cell_bounds[2], cell_bounds[1]),
                    (cell_bounds[2], cell_bounds[3]),
                    (cell_bounds[0], cell_bounds[3])
                ])
                
                # Count infrastructure in this cell
                wells_in_cell = wells_gdf[wells_gdf.geometry.within(cell_polygon)]
                
                # Calculate pipeline length in cell
                pipelines_in_cell = pipelines_gdf[pipelines_gdf.geometry.intersects(cell_polygon)]
                pipeline_length_km = 0
                if len(pipelines_in_cell) > 0:
                    for _, pipeline in pipelines_in_cell.iterrows():
                        intersection = pipeline.geometry.intersection(cell_polygon)
                        if intersection.length > 0:
                            # Convert to UTM for accurate length
                            intersection_utm = gpd.GeoSeries([intersection], crs='EPSG:4326').to_crs('EPSG:32612')
                            pipeline_length_km += intersection_utm.iloc[0].length / 1000  # m to km
                
                # Calculate risk metrics
                well_count = len(wells_in_cell)
                avg_well_risk = wells_in_cell['leak_risk_score'].mean() if well_count > 0 else 0
                
                avg_pipeline_risk = pipelines_in_cell['leak_risk_score'].mean() if len(pipelines_in_cell) > 0 else 0
                
                # Combined infrastructure density score
                well_density_score = well_count / (grid_size_km ** 2)  # wells per km²
                pipeline_density_score = pipeline_length_km / (grid_size_km ** 2)  # km of pipeline per km²
                
                # Weighted risk score
                total_infrastructure = well_count + pipeline_length_km
                if total_infrastructure > 0:
                    combined_risk_score = (
                        (well_count * avg_well_risk + pipeline_length_km * avg_pipeline_risk) /
                        total_infrastructure
                    )
                else:
                    combined_risk_score = 0
                
                grid_cells.append({
                    'grid_id': f"cell_{i}_{j}",
                    'x_center': (cell_bounds[0] + cell_bounds[2]) / 2,
                    'y_center': (cell_bounds[1] + cell_bounds[3]) / 2,
                    'well_count': well_count,
                    'pipeline_length_km': pipeline_length_km,
                    'well_density_per_km2': well_density_score,
                    'pipeline_density_km_per_km2': pipeline_density_score,
                    'avg_well_risk': avg_well_risk,
                    'avg_pipeline_risk': avg_pipeline_risk,
                    'combined_risk_score': combined_risk_score,
                    'total_infrastructure_score': well_density_score + pipeline_density_score,
                    'geometry': cell_polygon
                })
        
        density_gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        
        # Filter to cells with some infrastructure
        density_gdf = density_gdf[density_gdf['total_infrastructure_score'] > 0]
        
        self.logger.info(f"Calculated infrastructure density for {len(density_gdf)} grid cells")
        return density_gdf
    
    @log_function_call
    def process_all_infrastructure(self, well_sources: List, pipeline_sources: List,
                                 output_dir: Union[str, Path]) -> Dict[str, Union[Path, gpd.GeoDataFrame]]:
        """
        Process all infrastructure data and generate analysis products.
        
        Args:
            well_sources: List of well data sources
            pipeline_sources: List of pipeline data sources  
            output_dir: Directory for output files
            
        Returns:
            Dictionary with processed datasets and file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Processing all infrastructure data")
        
        # Process wells
        all_wells = []
        for source in well_sources:
            wells = self.load_well_data(source)
            all_wells.append(wells)
        
        if all_wells:
            combined_wells = gpd.pd.concat(all_wells, ignore_index=True)
        else:
            # Create sample data if no sources provided
            combined_wells = self.load_well_data(self.config.get('study_area', {}))
        
        # Process pipelines
        all_pipelines = []
        for source in pipeline_sources:
            pipelines = self.load_pipeline_data(source)
            all_pipelines.append(pipelines)
        
        if all_pipelines:
            combined_pipelines = gpd.pd.concat(all_pipelines, ignore_index=True)
        else:
            # Create sample data if no sources provided
            combined_pipelines = self.load_pipeline_data(self.config.get('study_area', {}))
        
        # Save raw infrastructure data
        wells_path = output_dir / "wells.geojson"
        pipelines_path = output_dir / "pipelines.geojson"
        
        combined_wells.to_file(wells_path, driver='GeoJSON')
        combined_pipelines.to_file(pipelines_path, driver='GeoJSON')
        
        # Create buffer zones
        buffers = self.create_infrastructure_buffers(combined_wells, combined_pipelines)
        
        # Save buffer zones
        buffer_paths = {}
        for buffer_name, buffer_gdf in buffers.items():
            buffer_path = output_dir / f"buffers_{buffer_name}.geojson"
            buffer_gdf.to_file(buffer_path, driver='GeoJSON')
            buffer_paths[buffer_name] = buffer_path
        
        # Calculate infrastructure density
        density_grid = self.calculate_infrastructure_density(
            combined_wells, combined_pipelines, 
            grid_size_km=1.0
        )
        
        density_path = output_dir / "infrastructure_density.geojson"
        density_grid.to_file(density_path, driver='GeoJSON')
        
        results = {
            'wells_data': combined_wells,
            'pipelines_data': combined_pipelines,
            'wells_path': wells_path,
            'pipelines_path': pipelines_path,
            'buffer_zones': buffers,
            'buffer_paths': buffer_paths,
            'density_grid': density_grid,
            'density_path': density_path,
            'well_count': len(combined_wells),
            'pipeline_count': len(combined_pipelines),
            'total_pipeline_length_km': combined_pipelines['length_miles'].sum() * 1.609344
        }
        
        self.logger.info("Completed processing all infrastructure data")
        return results