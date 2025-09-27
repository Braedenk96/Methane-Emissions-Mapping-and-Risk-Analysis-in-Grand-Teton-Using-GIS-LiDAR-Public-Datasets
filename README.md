# Methane Emissions Mapping and Risk Analysis in Grand Teton Using GIS, LiDAR & Public Datasets

A comprehensive spatial analysis workflow for methane emissions risk assessment that fuses multiple datasets including LiDAR, oil & gas infrastructure data, satellite-derived methane concentrations, and environmental layers to model and visualize methane emissions risk within and around Grand Teton National Park.

## 🎯 Project Overview

This project demonstrates advanced geospatial analysis capabilities by implementing a complete workflow for methane emissions risk assessment. The system integrates multiple data sources to create detailed risk maps and assessments that can support environmental monitoring and decision-making.

### Key Features

- **Multi-source Data Integration**: Combines LiDAR, satellite data, infrastructure databases, and environmental layers
- **Advanced Risk Modeling**: Sophisticated algorithms for proximity analysis, concentration assessment, and risk scoring
- **Interactive Visualizations**: Dynamic maps, statistical plots, and comprehensive reporting
- **Configurable Workflow**: Flexible configuration system for different study areas and analysis parameters
- **Production Ready**: Robust error handling, logging, and data validation throughout

### Business Applications

This workflow mirrors core capabilities needed for Gas Mapping LiDAR applications and showcases skills in:
- Independent problem-solving and system design
- Multi-source geospatial data fusion
- Emissions analysis and risk assessment
- Interactive visualization and reporting
- Production-ready code development

## 📊 Data Sources

### Primary Data Types
1. **LiDAR Point Clouds** - Terrain analysis and vegetation mapping
2. **Oil & Gas Infrastructure** - Wells, pipelines, and facilities data
3. **Satellite Methane Data** - TROPOMI/Sentinel-5P concentration measurements  
4. **Environmental Layers** - Elevation, land cover, meteorological data

### Supported Formats
- LiDAR: LAS/LAZ files
- Vector Data: Shapefiles, GeoJSON, GPKG
- Raster Data: GeoTIFF, NetCDF
- Tabular Data: CSV with coordinates

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Braedenk96/Methane-Emissions-Mapping-and-Risk-Analysis-in-Grand-Teton-Using-GIS-LiDAR-Public-Datasets.git
   cd Methane-Emissions-Mapping-and-Risk-Analysis-in-Grand-Teton-Using-GIS-LiDAR-Public-Datasets
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run demonstration with sample data**:
   ```bash
   python main.py --demo
   ```

### Basic Usage

Run the complete workflow with your own data:

```bash
python main.py \
    --lidar-sources path/to/lidar/*.las \
    --well-sources path/to/wells.shp \
    --pipeline-sources path/to/pipelines.shp \
    --satellite-sources path/to/methane_data.nc \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --grid-size-km 0.5
```

## 🏗️ Architecture

### Workflow Components

```
📁 src/
├── 🔧 utils/                    # Core utilities
│   ├── config.py               # Configuration management
│   ├── logging_utils.py        # Logging and progress tracking  
│   └── validation.py           # Data validation
├── 📊 data_processing/          # Data processing modules
│   ├── lidar_processor.py      # LiDAR point cloud processing
│   ├── infrastructure_processor.py  # Oil & gas infrastructure
│   └── satellite_processor.py  # Satellite methane data
├── 🧮 spatial_analysis/        # Analysis algorithms
│   └── risk_analyzer.py        # Risk assessment and modeling
└── 📈 visualization/           # Visualization and mapping
    └── interactive_mapping.py  # Interactive maps and plots
```

### Data Flow

1. **Data Ingestion** → Load and validate multiple data sources
2. **Data Processing** → Standardize, clean, and derive products
3. **Risk Analysis** → Calculate multi-factor risk scores
4. **Visualization** → Generate maps, plots, and reports
5. **Export** → Save results in multiple formats

## 📋 Configuration

The workflow uses a YAML configuration file (`config.yaml`) to manage:

- **Study Area**: Bounding box, coordinate system, buffer zones
- **Data Sources**: File paths, APIs, processing parameters
- **Risk Factors**: Weighting schemes, classification thresholds
- **Outputs**: Export formats, visualization options

Example configuration:
```yaml
study_area:
  name: "Grand Teton National Park"
  bounds:
    bbox: [-111.05, 43.65, -110.10, 43.85]
  epsg: 32612

analysis:
  risk_factors:
    weights:
      infrastructure_proximity: 0.35
      methane_concentration: 0.30
      terrain_factors: 0.15
      meteorological: 0.10
      landcover_sensitivity: 0.10
```

## 🎯 Risk Assessment Methodology

### Multi-Factor Risk Model

The risk assessment integrates five key factors:

1. **Infrastructure Proximity (35%)**
   - Distance-weighted risk from wells and pipelines
   - Infrastructure age and condition factors
   - Operational status considerations

2. **Methane Concentration (30%)**
   - Satellite-derived concentration anomalies
   - Temporal persistence of elevated levels
   - Background subtraction algorithms

3. **Terrain Factors (15%)**
   - Elevation and slope analysis from LiDAR DTMs
   - Wind flow and dispersion modeling
   - Topographic complexity metrics

4. **Meteorological (10%)**
   - Wind patterns and atmospheric stability
   - Temperature and pressure gradients
   - Seasonal variation factors

5. **Land Cover Sensitivity (10%)**
   - Environmental vulnerability assessment
   - Protected area proximity
   - Ecosystem impact potential

### Risk Classification

- **Very Low (0.0-0.2)**: Minimal emissions risk
- **Low (0.2-0.4)**: Limited concern areas  
- **Moderate (0.4-0.6)**: Standard monitoring recommended
- **High (0.6-0.8)**: Enhanced monitoring required
- **Very High (0.8-1.0)**: Immediate attention needed

## 📈 Outputs and Results

### Generated Products

1. **Interactive Risk Maps** - HTML maps with layered data visualization
2. **Risk Heatmaps** - Continuous risk surface visualization
3. **Statistical Reports** - Risk distribution analysis and correlations
4. **Infrastructure Analysis** - Well and pipeline risk assessments
5. **Temporal Analysis** - Methane concentration trends and patterns

### Export Formats

- **Geospatial**: GeoTIFF, Shapefile, GeoJSON, NetCDF
- **Visualization**: HTML (interactive), PNG (static plots)
- **Data**: CSV (tabular results), JSON (metadata)

### Sample Results Structure
```
📁 data/outputs/
├── 🗺️ visualizations/
│   ├── interactive_risk_map.html
│   ├── risk_heatmap.html
│   ├── risk_distribution_analysis.png
│   └── infrastructure_analysis.png
├── 📊 infrastructure/
│   ├── wells.geojson
│   ├── pipelines.geojson
│   └── infrastructure_density.geojson
├── 🛰️ satellite/
│   ├── processed_methane_data.nc
│   └── mean_methane_concentration.tif
└── 📄 analysis_summary.json
```

## 🔬 Technical Implementation

### Key Technologies
- **Python 3.8+** - Core development language
- **GeoPandas & Rasterio** - Geospatial data processing
- **xArray & NetCDF4** - Satellite data handling
- **Scikit-learn** - Machine learning for risk modeling
- **Folium & Plotly** - Interactive visualizations
- **Laspy & PDAL** - LiDAR point cloud processing

### Performance Considerations
- **Memory Management** - Chunked processing for large datasets
- **Parallel Processing** - Multi-threading for intensive operations
- **Progress Tracking** - Real-time feedback for long operations
- **Error Handling** - Robust validation and recovery mechanisms

## 📚 Documentation

### Example Notebooks
- `notebooks/01_data_exploration.ipynb` - Data source exploration
- `notebooks/02_risk_analysis_demo.ipynb` - Risk assessment walkthrough
- `notebooks/03_visualization_examples.ipynb` - Visualization gallery

### API Documentation
All classes and functions include comprehensive docstrings with:
- Parameter descriptions and types
- Return value specifications  
- Usage examples
- Error conditions

## 🧪 Testing and Validation

The workflow includes comprehensive validation:

- **Data Validation** - Format, projection, and quality checks
- **Spatial Validation** - Coordinate system and bounds verification
- **Statistical Validation** - Range and distribution checks
- **Integration Testing** - End-to-end workflow validation

## 🚀 Advanced Features

### Extensibility
- **Modular Design** - Easy addition of new data sources
- **Plugin Architecture** - Custom risk factors and algorithms
- **Configuration-Driven** - No code changes for new study areas

### Production Features
- **Batch Processing** - Handle multiple study areas
- **API Integration** - Connect to real-time data sources
- **Cloud Deployment** - Scalable processing infrastructure
- **Monitoring** - Health checks and performance metrics

## 🤝 Contributing

This project demonstrates professional software development practices:

1. **Code Organization** - Clean, modular architecture
2. **Documentation** - Comprehensive inline and external docs
3. **Configuration Management** - Flexible, version-controlled settings
4. **Error Handling** - Robust validation and logging
5. **Version Control** - Git best practices with clear commits

## 📞 Contact and Support

**Project Author**: Braeden K  
**Purpose**: Portfolio demonstration of geospatial analysis capabilities  
**Industry Focus**: Gas Mapping LiDAR and emissions monitoring applications

This project showcases advanced spatial analysis skills applicable to:
- Environmental monitoring and assessment
- Energy infrastructure risk analysis
- Remote sensing and satellite data processing
- Interactive visualization and reporting systems
- Production-ready geospatial workflows

---

**Note**: This is a portfolio project designed to demonstrate comprehensive geospatial analysis capabilities. The workflow includes both real data processing capabilities and sample data generation for demonstration purposes.