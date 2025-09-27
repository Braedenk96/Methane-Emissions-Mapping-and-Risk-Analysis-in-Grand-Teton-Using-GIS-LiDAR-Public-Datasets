# Example notebook demonstrating the methane emissions risk analysis workflow

This notebook provides a comprehensive demonstration of the methane emissions mapping and risk analysis system.

## Overview

The workflow integrates multiple geospatial datasets to assess methane emissions risk:
- LiDAR point cloud data for terrain analysis
- Oil & gas infrastructure data for proximity analysis  
- Satellite methane concentrations for emissions detection
- Environmental layers for sensitivity assessment

## Getting Started

First, let's import the necessary modules and set up the workflow:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path('..') / 'src'))

from src.utils import ConfigManager, Logger
from main import MethaneEmissionsWorkflow
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the workflow
workflow = MethaneEmissionsWorkflow('../config.yaml')

print("Methane Emissions Analysis Workflow - Demo")
print("==========================================")
```

## Step 1: Configuration Overview

```python
# Display configuration settings
config = workflow.config_manager.config

print("Study Area Configuration:")
print(f"  Region: {config['study_area']['name']}")
print(f"  Bounds: {config['study_area']['bounds']['bbox']}")
print(f"  CRS: EPSG:{config['study_area']['epsg']}")

print("\nRisk Factor Weights:")
weights = config['analysis']['risk_factors']['weights']
for factor, weight in weights.items():
    print(f"  {factor}: {weight}")
```

## Step 2: Run Complete Analysis

```python
# Run the complete workflow with sample data
print("Running complete analysis workflow...")
results = workflow.run_complete_workflow(grid_size_km=1.0)

print(f"Analysis completed successfully!")
print(f"Analysis grid contains {len(results['analysis_grid'])} cells")
```

## Step 3: Examine Risk Results

```python
# Display risk analysis statistics
if 'risk_statistics' in results:
    stats = results['risk_statistics']
    
    print("\nRisk Analysis Summary:")
    print(f"  Total analysis points: {stats['total_points']:,}")
    print(f"  Mean risk score: {stats['mean_risk_score']:.3f}")
    print(f"  Standard deviation: {stats['std_risk_score']:.3f}")
    print(f"  High priority areas: {stats['high_priority_points']}")
    
    print(f"\n Risk Category Distribution:")
    for category, count in stats['risk_distribution'].items():
        percentage = stats['risk_distribution_percent'][category]
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
```

## Step 4: Visualize Results

```python
# Create visualizations
if 'integrated_risk' in results:
    risk_df = results['integrated_risk']
    
    # Risk score distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(risk_df['composite_risk_score'], bins=30, alpha=0.7, color='skyblue')
    plt.title('Risk Score Distribution')
    plt.xlabel('Composite Risk Score')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    risk_counts = risk_df['risk_category'].value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Risk Category Distribution')
    
    plt.subplot(2, 2, 3)
    # Risk factor comparison
    risk_factors = ['infrastructure_risk_score', 'methane_concentration_risk', 'terrain_risk_score']
    available_factors = [f for f in risk_factors if f in risk_df.columns]
    if available_factors:
        risk_df[available_factors].boxplot()
        plt.title('Risk Factor Distributions')
        plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    # Scatter plot of top risk factors
    if len(available_factors) >= 2:
        plt.scatter(risk_df[available_factors[0]], risk_df[available_factors[1]], 
                   c=risk_df['composite_risk_score'], cmap='Reds', alpha=0.6)
        plt.colorbar(label='Risk Score')
        plt.xlabel(available_factors[0].replace('_', ' ').title())
        plt.ylabel(available_factors[1].replace('_', ' ').title())
        plt.title('Risk Factor Correlation')
    
    plt.tight_layout()
    plt.show()
```

## Step 5: Infrastructure Analysis

```python
# Analyze infrastructure data
if 'infrastructure_data' in results:
    infra = results['infrastructure_data']
    
    print("\nInfrastructure Analysis:")
    print(f"  Wells analyzed: {infra['well_count']}")
    print(f"  Pipelines analyzed: {infra['pipeline_count']}")
    print(f"  Total pipeline length: {infra['total_pipeline_length_km']:.1f} km")
    
    # Show wells data sample
    if 'wells_data' in infra and len(infra['wells_data']) > 0:
        wells_df = infra['wells_data']
        print(f"\nWell Data Sample:")
        print(wells_df[['well_id', 'well_type', 'status', 'leak_risk_score']].head())
```

## Step 6: Output Files

```python
# List generated output files
if 'visualization_results' in results:
    print("\nGenerated Visualizations:")
    for viz_type, file_path in results['visualization_results'].items():
        print(f"  {viz_type.replace('_', ' ').title()}: {file_path}")

if 'summary_path' in results:
    print(f"\nDetailed Summary: {results['summary_path']}")
```

## Key Insights

This demonstration showcases:

1. **Multi-source Data Integration**: The workflow successfully combines LiDAR, infrastructure, and satellite data
2. **Comprehensive Risk Assessment**: Risk scores incorporate multiple weighted factors
3. **Scalable Processing**: Grid-based analysis allows processing of large study areas
4. **Rich Visualizations**: Interactive maps and statistical plots provide multiple perspectives
5. **Production Ready**: Robust error handling and comprehensive logging throughout

## Next Steps

- Explore the generated HTML maps for interactive visualization
- Examine individual risk factors to understand their contributions
- Customize risk weights and thresholds for different applications
- Add real data sources for your specific study area

The workflow demonstrates advanced capabilities in:
- Geospatial data processing and analysis
- Multi-criteria risk assessment modeling
- Interactive visualization and reporting
- Production-ready software development practices