# Methane-Emissions-Mapping-and-Risk-Analysis-in-Grand-Teton-Using-GIS-LiDAR-Public-Datasets

Project Overview

Create a spatial analysis workflow that fuses multiple datasets — LiDAR, oil & gas infrastructure data, satellite-derived methane concentrations, and environmental layers — to model and visualize methane emissions risk within and around Grand Teton National Park.

Key Objectives
	1.	Map the oil & gas infrastructure near Grand Teton (pipelines, compressor stations, well pads, storage tanks).
	2.	Integrate methane emissions data from third-party sources (EPA GHGRP, satellite data from TROPOMI/EMIT, or state databases).
	3.	Fuse LiDAR or DEM data to identify terrain features that might influence plume movement or leak detection.
	4.	Create a risk index for emissions hotspots using proximity to infrastructure, wind patterns, and terrain.
	5.	Build an emissions dashboard/report that communicates findings visually to decision-makers (could be in ArcGIS Online, Tableau, or Python/Folium).

Responsibility
Creating workflows to generate actionable emissions information
-> Develop a reproducible pipeline to score and map methane emissions hotspots.
Integration and fusion of third-party data streams
-> Combine oil & gas infrastructure maps, satellite methane data, LiDAR DEMs, and weather layers.
Developing software tools for emissions analysis
-> Write Python scripts (or ArcPy/QGIS plugins) to automate data processing and reporting.
Understanding emissions regulations
-> Include layers showing regulated facilities or emission reporting thresholds (EPA GHGRP).
Generating and presenting emissions reports
-> Produce a polished, client-facing map and written summary explaining hotspots and risk.
Statistical analysis of large datasets
-> Use spatial statistics to quantify emission densities, clustering, and correlations.
Familiarity with oil and gas infrastructure
-. Leverage open data to map pipelines, storage tanks, and other assets.
Experience with GIS & SQL
-> Store and query your spatial data in a PostGIS or SQLite database.

Technical Stack To Use
	•	GIS Platform: QGIS or ArcGIS Pro (for visualization and geoprocessing)
	•	Python Libraries: geopandas, rasterio, shapely, folium, matplotlib, pandas
	•	Databases: PostGIS or SQLite (spatial queries, indexing)
	•	Data Fusion: Combine raster (satellite methane plumes) + vector (pipelines, facilities) + LiDAR DEM
	•	Reporting: Jupyter Notebook with interactive maps, or an ArcGIS Online dashboard

  Data Sources - Publicly Avaiable
	•	Oil & Gas Infrastructure:
	  •	Wyoming Oil & Gas Conservation Commission (WOGCC)
	  •	U.S. EIA Natural Gas Pipelines
	  •	EPA GHGRP Facility-Level Data
	•	Methane Concentrations:
	  •	TROPOMI CH₄ product from ESA Sentinel-5P
	  •	EMIT NASA Methane Plume Data
	•	LiDAR / DEM:
	  •	USGS 3D Elevation Program (3DEP)
	  •	Bridger Photonics public examples (if available)
	•	Environmental / Weather:
  	•	NOAA wind and temperature data

Deliverables
	1.	Interactive Map: Hotspots of potential methane emissions around Grand Teton.
	2.	Emissions Risk Score: Weighted index combining distance to infrastructure, terrain, and known emissions data.
	3.	Automated Workflow: Python or GIS ModelBuilder tool that can be rerun with updated data.
	4.	Professional Report: Written document summarizing findings, methodology, and policy implications.

  Optional Enhancements
	•	Model how wind direction and topography influence methane dispersion (simple Gaussian plume model or raster cost surface).
	•	Include regulatory overlays — facilities subject to specific EPA or state methane rules.
	•	Add time-series capability — show emissions hotspots over several months.
