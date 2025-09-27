"""
Interactive mapping and visualization module for methane emissions risk analysis.

This module provides tools for creating interactive maps, statistical plots,
and risk assessment visualizations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import contextily as ctx
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any
import warnings
import base64
import io

from ..utils import Logger, log_function_call


class InteractiveMapper:
    """Creates interactive maps for methane emissions risk visualization."""
    
    def __init__(self, config: Dict):
        """
        Initialize interactive mapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        
        # Color schemes for different risk levels
        self.risk_colors = {
            'very_low': '#2E7D32',      # Dark green
            'low': '#66BB6A',           # Light green  
            'moderate': '#FDD835',       # Yellow
            'high': '#FF8F00',          # Orange
            'very_high': '#D32F2F'      # Red
        }
        
        # Study area bounds
        self.study_bounds = config.get('study_area', {}).get('bounds', {}).get('bbox', 
                                     [-111.05, 43.65, -110.10, 43.85])
    
    @log_function_call
    def create_risk_map(self, risk_data: gpd.GeoDataFrame,
                       infrastructure_data: Dict[str, gpd.GeoDataFrame],
                       output_path: Union[str, Path]) -> Path:
        """
        Create interactive risk assessment map.
        
        Args:
            risk_data: GeoDataFrame with risk assessment results
            infrastructure_data: Dictionary with infrastructure datasets
            output_path: Output HTML file path
            
        Returns:
            Path to created map file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating interactive risk map: {output_path}")
        
        # Calculate map center
        if len(risk_data) > 0:
            center_lat = risk_data.geometry.centroid.y.mean()
            center_lon = risk_data.geometry.centroid.x.mean()
        else:
            center_lat = (self.study_bounds[1] + self.study_bounds[3]) / 2
            center_lon = (self.study_bounds[0] + self.study_bounds[2]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add additional tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
        
        # Add risk areas
        if len(risk_data) > 0 and 'risk_category' in risk_data.columns:
            for risk_level in ['very_low', 'low', 'moderate', 'high', 'very_high']:
                risk_subset = risk_data[risk_data['risk_category'] == risk_level]
                
                if len(risk_subset) > 0:
                    folium.GeoJson(
                        risk_subset.to_json(),
                        name=f'Risk Level: {risk_level.replace("_", " ").title()}',
                        style_function=lambda feature, color=self.risk_colors[risk_level]: {
                            'fillColor': color,
                            'color': color,
                            'weight': 2,
                            'fillOpacity': 0.7,
                        },
                        popup=folium.GeoJsonPopup(
                            fields=['composite_risk_score', 'risk_category', 'confidence_score'],
                            aliases=['Risk Score:', 'Risk Category:', 'Confidence:'],
                            localize=True,
                            labels=True
                        ),
                        tooltip=folium.GeoJsonTooltip(
                            fields=['composite_risk_score', 'risk_category'],
                            aliases=['Risk Score:', 'Category:'],
                            localize=True,
                            sticky=False,
                            labels=True
                        )
                    ).add_to(m)
        
        # Add infrastructure layers
        if 'wells_data' in infrastructure_data:
            wells_gdf = infrastructure_data['wells_data']
            if len(wells_gdf) > 0:
                # Wells layer
                well_markers = folium.FeatureGroup(name="Oil & Gas Wells")
                
                for idx, well in wells_gdf.iterrows():
                    # Color by risk level
                    risk_score = well.get('leak_risk_score', 0.5)
                    if risk_score > 0.8:
                        color = 'red'
                        icon = 'exclamation-triangle'
                    elif risk_score > 0.6:
                        color = 'orange'
                        icon = 'warning'
                    else:
                        color = 'blue'
                        icon = 'tint'
                    
                    popup_text = f"""
                    <b>Well ID:</b> {well.get('well_id', 'Unknown')}<br>
                    <b>Type:</b> {well.get('well_type', 'Unknown')}<br>
                    <b>Status:</b> {well.get('status', 'Unknown')}<br>
                    <b>Risk Score:</b> {risk_score:.2f}<br>
                    <b>Operator:</b> {well.get('operator', 'Unknown')}
                    """
                    
                    folium.Marker(
                        location=[well.geometry.y, well.geometry.x],
                        popup=folium.Popup(popup_text, max_width=200),
                        tooltip=f"Well: {well.get('well_id', 'Unknown')}",
                        icon=folium.Icon(color=color, icon=icon, prefix='fa')
                    ).add_to(well_markers)
                
                well_markers.add_to(m)
        
        if 'pipelines_data' in infrastructure_data:
            pipelines_gdf = infrastructure_data['pipelines_data']
            if len(pipelines_gdf) > 0:
                # Pipelines layer
                folium.GeoJson(
                    pipelines_gdf.to_json(),
                    name='Pipelines',
                    style_function=lambda feature: {
                        'color': '#FF6B35',
                        'weight': 3,
                        'opacity': 0.8
                    },
                    popup=folium.GeoJsonPopup(
                        fields=['pipeline_id', 'pipeline_type', 'operator', 'leak_risk_score'],
                        aliases=['Pipeline ID:', 'Type:', 'Operator:', 'Risk Score:'],
                        localize=True,
                        labels=True
                    )
                ).add_to(m)
        
        # Add buffer zones if available
        if 'buffer_zones' in infrastructure_data:
            buffer_zones = infrastructure_data['buffer_zones']
            
            # Add high-risk combined buffer
            if 'high_risk_combined' in buffer_zones:
                high_risk_buffer = buffer_zones['high_risk_combined']
                folium.GeoJson(
                    high_risk_buffer.to_json(),
                    name='High Risk Zones',
                    style_function=lambda feature: {
                        'fillColor': '#FF0000',
                        'color': '#FF0000',
                        'weight': 2,
                        'fillOpacity': 0.3,
                    }
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = self._create_legend_html()
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        # Add minimap
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Save map
        m.save(str(output_path))
        
        self.logger.info(f"Interactive risk map saved to {output_path}")
        return output_path
    
    def _create_legend_html(self) -> str:
        """Create HTML legend for the risk map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Levels</b></p>
        '''
        
        for risk_level, color in self.risk_colors.items():
            legend_html += f'''
            <p><span style="color:{color}">●</span> {risk_level.replace("_", " ").title()}</p>
            '''
        
        legend_html += '</div>'
        return legend_html
    
    @log_function_call
    def create_heatmap(self, risk_data: gpd.GeoDataFrame, 
                      output_path: Union[str, Path],
                      value_column: str = 'composite_risk_score') -> Path:
        """
        Create risk heatmap visualization.
        
        Args:
            risk_data: GeoDataFrame with risk assessment results
            output_path: Output HTML file path  
            value_column: Column to use for heatmap values
            
        Returns:
            Path to created heatmap file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating risk heatmap: {output_path}")
        
        # Calculate center
        if len(risk_data) > 0:
            center_lat = risk_data.geometry.centroid.y.mean()
            center_lon = risk_data.geometry.centroid.x.mean()
        else:
            center_lat = (self.study_bounds[1] + self.study_bounds[3]) / 2
            center_lon = (self.study_bounds[0] + self.study_bounds[2]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB positron'
        )
        
        # Prepare heatmap data
        if len(risk_data) > 0 and value_column in risk_data.columns:
            heat_data = []
            
            for idx, row in risk_data.iterrows():
                if not pd.isna(row[value_column]):
                    lat = row.geometry.centroid.y
                    lon = row.geometry.centroid.x
                    weight = float(row[value_column])
                    heat_data.append([lat, lon, weight])
            
            if heat_data:
                # Add heatmap layer
                plugins.HeatMap(
                    heat_data,
                    name='Risk Heatmap',
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=25,
                    blur=15,
                    gradient={
                        0.0: 'green',
                        0.3: 'yellow', 
                        0.7: 'orange',
                        1.0: 'red'
                    }
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save heatmap
        m.save(str(output_path))
        
        self.logger.info(f"Risk heatmap saved to {output_path}")
        return output_path


class StatisticalPlotter:
    """Creates statistical plots and charts for risk analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize statistical plotter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    @log_function_call
    def create_risk_distribution_plot(self, risk_data: pd.DataFrame,
                                    output_path: Union[str, Path]) -> Path:
        """
        Create risk score distribution plots.
        
        Args:
            risk_data: DataFrame with risk assessment results
            output_path: Output file path
            
        Returns:
            Path to created plot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating risk distribution plot: {output_path}")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Methane Emissions Risk Analysis - Statistical Distribution', fontsize=16, fontweight='bold')
        
        # Risk score histogram
        if 'composite_risk_score' in risk_data.columns:
            axes[0, 0].hist(risk_data['composite_risk_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Risk Score Distribution')
            axes[0, 0].set_xlabel('Composite Risk Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add vertical lines for risk thresholds
            thresholds = self.config.get('analysis', {}).get('risk_classification', {}).get('thresholds', {})
            for threshold_name, threshold_value in thresholds.items():
                axes[0, 0].axvline(threshold_value, color='red', linestyle='--', alpha=0.7, 
                                 label=f'{threshold_name}: {threshold_value}')
        
        # Risk category pie chart
        if 'risk_category' in risk_data.columns:
            risk_counts = risk_data['risk_category'].value_counts()
            colors = [self._get_risk_color(cat) for cat in risk_counts.index]
            
            wedges, texts, autotexts = axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, 
                                                    autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0, 1].set_title('Risk Category Distribution')
            
            # Beautify text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # Box plot of risk factors
        risk_factors = ['infrastructure_risk_score', 'methane_concentration_risk', 
                       'terrain_risk_score', 'sensitivity_score']
        
        available_factors = [f for f in risk_factors if f in risk_data.columns]
        if available_factors:
            risk_factor_data = risk_data[available_factors].melt()
            sns.boxplot(data=risk_factor_data, x='variable', y='value', ax=axes[1, 0])
            axes[1, 0].set_title('Risk Factor Distributions')
            axes[1, 0].set_xlabel('Risk Factors')
            axes[1, 0].set_ylabel('Risk Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        numeric_columns = risk_data.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_columns if 'risk' in col.lower() or 'score' in col.lower()]
        
        if len(correlation_cols) > 1:
            correlation_matrix = risk_data[correlation_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 1], fmt='.2f')
            axes[1, 1].set_title('Risk Factor Correlations')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Risk distribution plot saved to {output_path}")
        return output_path
    
    @log_function_call
    def create_infrastructure_analysis_plot(self, infrastructure_data: Dict,
                                          output_path: Union[str, Path]) -> Path:
        """
        Create infrastructure analysis plots.
        
        Args:
            infrastructure_data: Dictionary with infrastructure datasets
            output_path: Output file path
            
        Returns:
            Path to created plot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating infrastructure analysis plot: {output_path}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Infrastructure Analysis for Methane Emissions Risk', fontsize=16, fontweight='bold')
        
        # Wells analysis
        if 'wells_data' in infrastructure_data and len(infrastructure_data['wells_data']) > 0:
            wells_df = infrastructure_data['wells_data']
            
            # Well types distribution
            if 'well_type' in wells_df.columns:
                well_type_counts = wells_df['well_type'].value_counts()
                axes[0, 0].pie(well_type_counts.values, labels=well_type_counts.index, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Well Types Distribution')
            
            # Well risk scores
            if 'leak_risk_score' in wells_df.columns:
                axes[0, 1].hist(wells_df['leak_risk_score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[0, 1].set_title('Well Risk Score Distribution')
                axes[0, 1].set_xlabel('Leak Risk Score')
                axes[0, 1].set_ylabel('Number of Wells')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Pipeline analysis
        if 'pipelines_data' in infrastructure_data and len(infrastructure_data['pipelines_data']) > 0:
            pipelines_df = infrastructure_data['pipelines_data']
            
            # Pipeline types
            if 'pipeline_type' in pipelines_df.columns:
                pipe_type_counts = pipelines_df['pipeline_type'].value_counts()
                axes[1, 0].bar(pipe_type_counts.index, pipe_type_counts.values, color='lightblue', edgecolor='black')
                axes[1, 0].set_title('Pipeline Types')
                axes[1, 0].set_xlabel('Pipeline Type')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Pipeline length vs risk
            if 'length_miles' in pipelines_df.columns and 'leak_risk_score' in pipelines_df.columns:
                scatter = axes[1, 1].scatter(pipelines_df['length_miles'], pipelines_df['leak_risk_score'], 
                                          alpha=0.6, c=pipelines_df['leak_risk_score'], cmap='Reds')
                axes[1, 1].set_title('Pipeline Length vs Risk Score')
                axes[1, 1].set_xlabel('Length (miles)')
                axes[1, 1].set_ylabel('Leak Risk Score')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Infrastructure analysis plot saved to {output_path}")
        return output_path
    
    @log_function_call
    def create_temporal_analysis_plot(self, methane_data: pd.DataFrame,
                                    output_path: Union[str, Path]) -> Path:
        """
        Create temporal analysis plots for methane data.
        
        Args:
            methane_data: DataFrame with temporal methane data
            output_path: Output file path
            
        Returns:
            Path to created plot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating temporal analysis plot: {output_path}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Methane Concentration Analysis', fontsize=16, fontweight='bold')
        
        # Time series plot (if time column exists)
        if 'time' in methane_data.columns and 'methane_concentration' in methane_data.columns:
            methane_data['time'] = pd.to_datetime(methane_data['time'])
            
            # Monthly average
            monthly_avg = methane_data.set_index('time')['methane_concentration'].resample('M').mean()
            axes[0, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Monthly Average Methane Concentration')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('CH₄ Concentration (ppb)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Seasonal patterns
        if 'time' in methane_data.columns:
            methane_data['month'] = pd.to_datetime(methane_data['time']).dt.month
            if 'methane_concentration' in methane_data.columns:
                seasonal_avg = methane_data.groupby('month')['methane_concentration'].mean()
                axes[0, 1].bar(seasonal_avg.index, seasonal_avg.values, color='lightgreen', edgecolor='black')
                axes[0, 1].set_title('Seasonal Methane Patterns')
                axes[0, 1].set_xlabel('Month')
                axes[0, 1].set_ylabel('Average CH₄ (ppb)')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Anomaly distribution
        if 'methane_anomaly' in methane_data.columns:
            axes[1, 0].hist(methane_data['methane_anomaly'], bins=30, alpha=0.7, 
                          color='salmon', edgecolor='black')
            axes[1, 0].set_title('Methane Anomaly Distribution')
            axes[1, 0].set_xlabel('Anomaly (ppb)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add vertical line at zero
            axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Concentration vs anomaly scatter
        if 'methane_concentration' in methane_data.columns and 'methane_anomaly' in methane_data.columns:
            scatter = axes[1, 1].scatter(methane_data['methane_concentration'], 
                                       methane_data['methane_anomaly'],
                                       alpha=0.5, c=methane_data['methane_anomaly'], 
                                       cmap='RdYlBu_r')
            axes[1, 1].set_title('Concentration vs Anomaly')
            axes[1, 1].set_xlabel('CH₄ Concentration (ppb)')
            axes[1, 1].set_ylabel('CH₄ Anomaly (ppb)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Temporal analysis plot saved to {output_path}")
        return output_path
    
    def _get_risk_color(self, risk_category: str) -> str:
        """Get color for risk category."""
        color_map = {
            'very_low': '#2E7D32',
            'low': '#66BB6A',
            'moderate': '#FDD835',
            'high': '#FF8F00',
            'very_high': '#D32F2F'
        }
        return color_map.get(risk_category, '#808080')


class RiskMapper:
    """Main class for coordinating all visualization components."""
    
    def __init__(self, config: Dict):
        """
        Initialize risk mapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_logger(__name__)
        
        self.interactive_mapper = InteractiveMapper(config)
        self.statistical_plotter = StatisticalPlotter(config)
    
    @log_function_call
    def create_comprehensive_visualization(self, analysis_results: Dict[str, Any],
                                         output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Create comprehensive visualization suite.
        
        Args:
            analysis_results: Dictionary with all analysis results
            output_dir: Output directory for visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating comprehensive visualization suite in {output_dir}")
        
        visualization_outputs = {}
        
        # Interactive risk map
        if 'risk_assessment' in analysis_results:
            risk_map_path = self.interactive_mapper.create_risk_map(
                analysis_results['risk_assessment'],
                analysis_results.get('infrastructure_data', {}),
                output_dir / 'interactive_risk_map.html'
            )
            visualization_outputs['interactive_map'] = risk_map_path
            
            # Risk heatmap
            heatmap_path = self.interactive_mapper.create_heatmap(
                analysis_results['risk_assessment'],
                output_dir / 'risk_heatmap.html'
            )
            visualization_outputs['risk_heatmap'] = heatmap_path
        
        # Statistical plots
        if 'integrated_risk' in analysis_results:
            risk_dist_path = self.statistical_plotter.create_risk_distribution_plot(
                analysis_results['integrated_risk'],
                output_dir / 'risk_distribution_analysis.png'
            )
            visualization_outputs['risk_distribution'] = risk_dist_path
        
        if 'infrastructure_data' in analysis_results:
            infra_plot_path = self.statistical_plotter.create_infrastructure_analysis_plot(
                analysis_results['infrastructure_data'],
                output_dir / 'infrastructure_analysis.png'
            )
            visualization_outputs['infrastructure_analysis'] = infra_plot_path
        
        # Temporal analysis (if available)
        if 'methane_data' in analysis_results:
            # Convert xarray Dataset to DataFrame for plotting
            methane_ds = analysis_results['methane_data']
            if hasattr(methane_ds, 'to_dataframe'):
                try:
                    methane_df = methane_ds.to_dataframe().reset_index()
                    temporal_plot_path = self.statistical_plotter.create_temporal_analysis_plot(
                        methane_df,
                        output_dir / 'temporal_methane_analysis.png'
                    )
                    visualization_outputs['temporal_analysis'] = temporal_plot_path
                except Exception as e:
                    self.logger.warning(f"Could not create temporal analysis plot: {e}")
        
        self.logger.info(f"Created {len(visualization_outputs)} visualizations")
        return visualization_outputs