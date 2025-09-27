#!/usr/bin/env python3
"""
Main workflow script for methane emissions mapping and risk analysis.

This script orchestrates the entire analysis workflow, from data processing
through risk assessment to visualization and reporting.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import geopandas as gpd
import numpy as np
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import ConfigManager, EnvironmentManager, Logger, log_function_call
from src.data_processing import LiDARProcessor, InfrastructureProcessor, SatelliteDataProcessor
from src.spatial_analysis import RiskAnalyzer
from src.visualization import RiskMapper


class MethaneEmissionsWorkflow:
    """Main workflow orchestrator for methane emissions risk analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the workflow.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Setup environment
        EnvironmentManager.setup_directories()
        
        # Initialize logger
        self.logger = Logger()
        self.logger = Logger.get_logger(__name__)
        
        # Initialize processors
        self.lidar_processor = LiDARProcessor(self.config)
        self.infrastructure_processor = InfrastructureProcessor(self.config)
        self.satellite_processor = SatelliteDataProcessor(self.config)
        self.risk_analyzer = RiskAnalyzer(self.config)
        self.risk_mapper = RiskMapper(self.config)
        
        self.logger.info("Methane Emissions Workflow initialized")
    
    @log_function_call
    def create_analysis_grid(self, grid_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """
        Create analysis grid for the study area.
        
        Args:
            grid_size_km: Grid cell size in kilometers
            
        Returns:
            GeoDataFrame with analysis grid
        """
        self.logger.info(f"Creating analysis grid with {grid_size_km}km resolution")
        
        # Get study area bounds
        bounds = self.config_manager.get_study_area_bounds()
        minx, miny, maxx, maxy = bounds
        
        # Convert km to degrees (rough approximation)
        grid_size_deg = grid_size_km / 111  # ~111 km per degree
        
        # Create grid
        x_coords = np.arange(minx, maxx + grid_size_deg, grid_size_deg)
        y_coords = np.arange(miny, maxy + grid_size_deg, grid_size_deg)
        
        grid_cells = []
        grid_id = 0
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                # Create grid cell polygon
                from shapely.geometry import Polygon
                
                cell_polygon = Polygon([
                    (x, y),
                    (x_coords[i + 1], y),
                    (x_coords[i + 1], y_coords[j + 1]),
                    (x, y_coords[j + 1])
                ])
                
                grid_cells.append({
                    'grid_id': grid_id,
                    'x_center': (x + x_coords[i + 1]) / 2,
                    'y_center': (y + y_coords[j + 1]) / 2,
                    'geometry': cell_polygon
                })
                
                grid_id += 1
        
        analysis_grid = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        
        self.logger.info(f"Created analysis grid with {len(analysis_grid)} cells")
        return analysis_grid
    
    @log_function_call
    def process_lidar_data(self, lidar_sources: List[str]) -> Dict[str, Any]:
        """
        Process LiDAR data sources.
        
        Args:
            lidar_sources: List of LiDAR data source paths
            
        Returns:
            Dictionary with processed LiDAR results
        """
        self.logger.info("Processing LiDAR data")
        
        output_dir = EnvironmentManager.get_output_dir() / "lidar"
        lidar_results = {}
        
        if not lidar_sources:
            self.logger.info("No LiDAR sources provided - skipping LiDAR processing")
            return lidar_results
        
        try:
            for lidar_source in lidar_sources:
                source_path = Path(lidar_source)
                if source_path.exists():
                    result = self.lidar_processor.process_lidar_tile(source_path, output_dir)
                    lidar_results[source_path.stem] = result
                else:
                    self.logger.warning(f"LiDAR source not found: {lidar_source}")
            
            if lidar_results:
                self.logger.info(f"Processed {len(lidar_results)} LiDAR tiles")
            else:
                self.logger.warning("No LiDAR data could be processed")
        
        except Exception as e:
            self.logger.error(f"Error processing LiDAR data: {e}")
        
        return lidar_results
    
    @log_function_call 
    def process_infrastructure_data(self, well_sources: List[str], 
                                  pipeline_sources: List[str]) -> Dict[str, Any]:
        """
        Process infrastructure data sources.
        
        Args:
            well_sources: List of well data sources
            pipeline_sources: List of pipeline data sources
            
        Returns:
            Dictionary with processed infrastructure results
        """
        self.logger.info("Processing infrastructure data")
        
        output_dir = EnvironmentManager.get_output_dir() / "infrastructure"
        
        try:
            # If no sources provided, use sample data
            if not well_sources and not pipeline_sources:
                self.logger.info("No infrastructure sources provided - using sample data")
                well_sources = [self.config.get('study_area', {})]
                pipeline_sources = [self.config.get('study_area', {})]
            
            infrastructure_results = self.infrastructure_processor.process_all_infrastructure(
                well_sources, pipeline_sources, output_dir
            )
            
            self.logger.info("Infrastructure data processing completed")
            return infrastructure_results
        
        except Exception as e:
            self.logger.error(f"Error processing infrastructure data: {e}")
            return {}
    
    @log_function_call
    def process_satellite_data(self, satellite_sources: List[str],
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process satellite methane data.
        
        Args:
            satellite_sources: List of satellite data sources
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with processed satellite results
        """
        self.logger.info("Processing satellite methane data")
        
        output_dir = EnvironmentManager.get_output_dir() / "satellite"
        
        try:
            # If no sources provided, use sample data
            if not satellite_sources:
                self.logger.info("No satellite sources provided - using sample data")
                satellite_sources = [self.config.get('study_area', {})]
            
            satellite_results = self.satellite_processor.process_satellite_data(
                satellite_sources, output_dir, start_date, end_date
            )
            
            self.logger.info("Satellite data processing completed")
            return satellite_results
        
        except Exception as e:
            self.logger.error(f"Error processing satellite data: {e}")
            return {}
    
    @log_function_call
    def perform_risk_analysis(self, analysis_grid: gpd.GeoDataFrame,
                            infrastructure_data: Dict[str, Any],
                            satellite_data: Dict[str, Any],
                            lidar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            analysis_grid: Analysis grid for assessment
            infrastructure_data: Processed infrastructure data
            satellite_data: Processed satellite data
            lidar_data: Processed LiDAR data
            
        Returns:
            Dictionary with risk analysis results
        """
        self.logger.info("Performing comprehensive risk analysis")
        
        try:
            # Calculate infrastructure proximity risk
            wells_gdf = infrastructure_data.get('wells_data', gpd.GeoDataFrame())
            pipelines_gdf = infrastructure_data.get('pipelines_data', gpd.GeoDataFrame())
            buffer_zones = infrastructure_data.get('buffer_zones', {})
            
            infrastructure_risk = self.risk_analyzer.calculate_infrastructure_risk(
                analysis_grid, wells_gdf, pipelines_gdf, buffer_zones
            )
            
            # Calculate methane concentration risk
            methane_dataset = satellite_data.get('full_dataset')
            if methane_dataset is not None:
                methane_risk = self.risk_analyzer.calculate_methane_concentration_risk(
                    analysis_grid, methane_dataset
                )
            else:
                # Create empty methane risk if no data available
                self.logger.warning("No methane dataset available - using default risk scores")
                methane_risk = infrastructure_risk[['point_id']].copy()
                methane_risk['methane_concentration_risk'] = 0.1
            
            # Calculate terrain risk (using LiDAR DTM if available)
            dtm_path = None
            if lidar_data:
                # Find first DTM file
                for tile_result in lidar_data.values():
                    if 'dtm' in tile_result:
                        dtm_path = tile_result['dtm']
                        break
            
            terrain_risk = self.risk_analyzer.calculate_terrain_risk(
                analysis_grid, dtm_path
            )
            
            # Calculate land cover sensitivity (would need land cover data)
            landcover_risk = self.risk_analyzer.calculate_landcover_sensitivity(
                analysis_grid
            )
            
            # Integrate all risk factors
            integrated_risk = self.risk_analyzer.integrate_risk_factors(
                infrastructure_risk, methane_risk, terrain_risk, landcover_risk
            )
            
            # Generate risk statistics
            risk_statistics = self.risk_analyzer.generate_risk_statistics(integrated_risk)
            
            # Add geometry to integrated risk for visualization
            integrated_risk_gdf = analysis_grid.merge(
                integrated_risk, left_index=True, right_on='point_id', how='inner'
            )
            
            risk_results = {
                'infrastructure_risk': infrastructure_risk,
                'methane_risk': methane_risk,
                'terrain_risk': terrain_risk,
                'landcover_risk': landcover_risk,
                'integrated_risk': integrated_risk,
                'risk_assessment': integrated_risk_gdf,
                'risk_statistics': risk_statistics
            }
            
            self.logger.info("Risk analysis completed successfully")
            return risk_results
        
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}
    
    @log_function_call
    def create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create visualizations and reports.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Dictionary with visualization file paths
        """
        self.logger.info("Creating visualizations and reports")
        
        output_dir = EnvironmentManager.get_output_dir() / "visualizations"
        
        try:
            visualization_results = self.risk_mapper.create_comprehensive_visualization(
                analysis_results, output_dir
            )
            
            self.logger.info(f"Created {len(visualization_results)} visualizations")
            return visualization_results
        
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return {}
    
    @log_function_call
    def save_results_summary(self, analysis_results: Dict[str, Any]) -> Path:
        """
        Save a comprehensive results summary.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Path to summary file
        """
        output_dir = EnvironmentManager.get_output_dir()
        summary_path = output_dir / "analysis_summary.json"
        
        # Create serializable summary
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'data_processing_summary': {},
            'risk_analysis_summary': {},
            'file_outputs': {}
        }
        
        # Add data processing summaries
        if 'infrastructure_data' in analysis_results:
            infra_data = analysis_results['infrastructure_data']
            summary['data_processing_summary']['infrastructure'] = {
                'wells_count': infra_data.get('well_count', 0),
                'pipelines_count': infra_data.get('pipeline_count', 0),
                'total_pipeline_length_km': infra_data.get('total_pipeline_length_km', 0)
            }
        
        if 'satellite_data' in analysis_results:
            sat_data = analysis_results['satellite_data']
            summary['data_processing_summary']['satellite'] = {
                'total_observations': sat_data.get('total_observations', 0),
                'temporal_extent': sat_data.get('temporal_extent', {}),
                'spatial_extent': sat_data.get('spatial_extent', {})
            }
        
        # Add risk analysis summary
        if 'risk_statistics' in analysis_results:
            risk_stats = analysis_results['risk_statistics']
            # Convert numpy types to native Python types for JSON serialization
            serializable_stats = {}
            for key, value in risk_stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, dict):
                    serializable_stats[key] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                                             for k, v in value.items()}
                else:
                    serializable_stats[key] = value
            
            summary['risk_analysis_summary'] = serializable_stats
        
        # Add file output paths
        if 'visualization_results' in analysis_results:
            vis_results = analysis_results['visualization_results']
            summary['file_outputs']['visualizations'] = {
                key: str(path) for key, path in vis_results.items()
            }
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Analysis summary saved to {summary_path}")
        return summary_path
    
    @log_function_call
    def run_complete_workflow(self, 
                            lidar_sources: Optional[List[str]] = None,
                            well_sources: Optional[List[str]] = None,
                            pipeline_sources: Optional[List[str]] = None,
                            satellite_sources: Optional[List[str]] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            grid_size_km: float = 1.0) -> Dict[str, Any]:
        """
        Run the complete methane emissions analysis workflow.
        
        Args:
            lidar_sources: List of LiDAR data sources
            well_sources: List of well data sources
            pipeline_sources: List of pipeline data sources
            satellite_sources: List of satellite data sources
            start_date: Analysis start date
            end_date: Analysis end date
            grid_size_km: Analysis grid resolution
            
        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info("Starting complete methane emissions analysis workflow")
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Step 1: Create analysis grid
            analysis_grid = self.create_analysis_grid(grid_size_km)
            results['analysis_grid'] = analysis_grid
            
            # Step 2: Process data sources
            self.logger.info("=== DATA PROCESSING PHASE ===")
            
            # Process LiDAR data
            lidar_results = self.process_lidar_data(lidar_sources or [])
            results['lidar_data'] = lidar_results
            
            # Process infrastructure data  
            infrastructure_results = self.process_infrastructure_data(
                well_sources or [], pipeline_sources or []
            )
            results['infrastructure_data'] = infrastructure_results
            
            # Process satellite data
            satellite_results = self.process_satellite_data(
                satellite_sources or [], start_date, end_date
            )
            results['satellite_data'] = satellite_results
            
            # Step 3: Risk analysis
            self.logger.info("=== RISK ANALYSIS PHASE ===")
            
            risk_results = self.perform_risk_analysis(
                analysis_grid, infrastructure_results, satellite_results, lidar_results
            )
            results.update(risk_results)
            
            # Step 4: Visualization and reporting
            self.logger.info("=== VISUALIZATION PHASE ===")
            
            visualization_results = self.create_visualizations(results)
            results['visualization_results'] = visualization_results
            
            # Step 5: Save summary
            summary_path = self.save_results_summary(results)
            results['summary_path'] = summary_path
            
            self.logger.info("=== WORKFLOW COMPLETED SUCCESSFULLY ===")
            
            # Print summary to console
            self._print_workflow_summary(results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Workflow failed with error: {e}")
            raise
    
    def _print_workflow_summary(self, results: Dict[str, Any]):
        """Print workflow summary to console."""
        print("\n" + "="*60)
        print("METHANE EMISSIONS ANALYSIS - WORKFLOW SUMMARY")
        print("="*60)
        
        # Data processing summary
        if 'infrastructure_data' in results:
            infra = results['infrastructure_data']
            print(f"\n📍 INFRASTRUCTURE DATA:")
            print(f"   • Wells processed: {infra.get('well_count', 0)}")
            print(f"   • Pipelines processed: {infra.get('pipeline_count', 0)}")
            print(f"   • Total pipeline length: {infra.get('total_pipeline_length_km', 0):.1f} km")
        
        if 'satellite_data' in results:
            sat = results['satellite_data']
            print(f"\n🛰️  SATELLITE DATA:")
            print(f"   • Total observations: {sat.get('total_observations', 0):,}")
            temporal = sat.get('temporal_extent', {})
            if temporal:
                print(f"   • Temporal range: {temporal.get('start', 'N/A')} to {temporal.get('end', 'N/A')}")
        
        if 'lidar_data' in results and results['lidar_data']:
            print(f"\n📊 LIDAR DATA:")
            print(f"   • Tiles processed: {len(results['lidar_data'])}")
        
        # Risk analysis summary
        if 'risk_statistics' in results:
            risk_stats = results['risk_statistics']
            print(f"\n⚠️  RISK ANALYSIS:")
            print(f"   • Analysis points: {risk_stats.get('total_points', 0):,}")
            print(f"   • Mean risk score: {risk_stats.get('mean_risk_score', 0):.3f}")
            print(f"   • High priority areas: {risk_stats.get('high_priority_points', 0)}")
            
            risk_dist = risk_stats.get('risk_distribution', {})
            if risk_dist:
                print(f"   • Risk distribution:")
                for risk_level, count in risk_dist.items():
                    print(f"     - {risk_level.replace('_', ' ').title()}: {count}")
        
        # Output files
        if 'visualization_results' in results:
            vis_results = results['visualization_results']
            print(f"\n📈 VISUALIZATIONS CREATED:")
            for vis_type, path in vis_results.items():
                print(f"   • {vis_type.replace('_', ' ').title()}: {path}")
        
        if 'summary_path' in results:
            print(f"\n📄 SUMMARY REPORT: {results['summary_path']}")
        
        print("\n" + "="*60)
        print("Analysis complete! Check the output directory for all results.")
        print("="*60 + "\n")


def main():
    """Main entry point for the workflow script."""
    parser = argparse.ArgumentParser(
        description="Methane Emissions Mapping and Risk Analysis Workflow"
    )
    
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--lidar-sources",
        nargs="*",
        help="Paths to LiDAR data sources"
    )
    
    parser.add_argument(
        "--well-sources", 
        nargs="*",
        help="Paths to well data sources"
    )
    
    parser.add_argument(
        "--pipeline-sources",
        nargs="*", 
        help="Paths to pipeline data sources"
    )
    
    parser.add_argument(
        "--satellite-sources",
        nargs="*",
        help="Paths to satellite data sources"
    )
    
    parser.add_argument(
        "--start-date",
        help="Analysis start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", 
        help="Analysis end date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--grid-size-km",
        type=float,
        default=1.0,
        help="Analysis grid size in kilometers"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample data"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize workflow
        workflow = MethaneEmissionsWorkflow(args.config)
        
        if args.demo:
            # Run demonstration with sample data
            print("Running demonstration with sample data...")
            results = workflow.run_complete_workflow(
                grid_size_km=args.grid_size_km
            )
        else:
            # Run with provided data sources
            results = workflow.run_complete_workflow(
                lidar_sources=args.lidar_sources,
                well_sources=args.well_sources,
                pipeline_sources=args.pipeline_sources,
                satellite_sources=args.satellite_sources,
                start_date=args.start_date,
                end_date=args.end_date,
                grid_size_km=args.grid_size_km
            )
        
        return 0
    
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())