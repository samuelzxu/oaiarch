# Sentinel-2 STAC Data Downloader

A fast and efficient Python script for downloading Sentinel-2 L2A satellite data using STAC (SpatioTemporal Asset Catalog) for any lat/lon bounding box.

## Features

- ✅ **No authentication required** - Uses public STAC catalogs
- ✅ **Fast downloads** - Only downloads specific bands, not entire products
- ✅ **Direct data processing** - Loads data into xarray for immediate analysis
- ✅ **Automatic scaling** - Applies scale/offset corrections
- ✅ **Vegetation indices** - Calculates NDVI, NDWI, EVI automatically
- ✅ **Cloud-optimized** - Uses COG (Cloud Optimized GeoTIFF) format

## Target Bands

- **B02 (Blue)**: 490nm - 10m resolution
- **B03 (Green)**: 560nm - 10m resolution  
- **B04 (Red)**: 665nm - 10m resolution
- **B08 (NIR)**: 842nm - 10m resolution
- **TCI (Visual)**: True Color Image - RGB composite

## Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install pystac-client odc-stac rasterio xarray pandas numpy
```

## Usage

### Basic Usage

```python
from query_sentinel_data import SentinelSTACDownloader

# Initialize downloader (no credentials needed!)
downloader = SentinelSTACDownloader()

# Define your area of interest
north, south = 37.9, 37.3  # San Francisco Bay Area
east, west = -122.0, -122.8

# Search for products
search_results = downloader.search_items(
    north=north, south=south, east=east, west=west,
    start_date="2024-01-01", end_date="2024-01-31",
    cloud_cover_max=20
)

# Load data directly into xarray
geometry = downloader.create_geometry(north, south, east, west)
data = downloader.load_data(search_results, geometry=geometry)

# Apply scaling and calculate indices
scaled_data = downloader.apply_scaling(data, search_results)
final_data = downloader.calculate_indices(scaled_data)

# Save as GeoTIFF files
saved_files = downloader.save_data(final_data, "output_directory")
```

### Run the Main Script

```bash
python query_sentinel_data.py
```

Edit the configuration section in `main()` to set your area of interest:

```python
# Bounding box coordinates
NORTH = 37.9   # Northern latitude
SOUTH = 37.3   # Southern latitude
EAST = -122.0  # Eastern longitude  
WEST = -122.8  # Western longitude

# Search parameters
START_DATE = "2024-01-01"
END_DATE = "2024-01-31"
MAX_CLOUD_COVER = 20
```

### Multiple Regions Example

```bash
python example_usage.py
```

This will download data for multiple predefined regions including San Francisco, Los Angeles, New York City, and London.

## Output

The script creates:

1. **GeoTIFF files** for each band and date:
   - `sentinel2_l2a_blue_20240115.tif`
   - `sentinel2_l2a_green_20240115.tif`
   - `sentinel2_l2a_red_20240115.tif`
   - `sentinel2_l2a_nir_20240115.tif`
   - `sentinel2_l2a_visual_20240115.tif`

2. **Calculated indices**:
   - `sentinel2_l2a_ndvi_20240115.tif` (Vegetation)
   - `sentinel2_l2a_ndwi_20240115.tif` (Water)
   - `sentinel2_l2a_evi_20240115.tif` (Enhanced Vegetation)

3. **Metadata file**: `sentinel2_l2a_metadata.json`

## Advantages over Traditional Methods

| Feature | STAC Method | Traditional (sentinelsat) |
|---------|-------------|---------------------------|
| Authentication | ❌ None required | ✅ Required |
| Download speed | ⚡ Very fast | 🐌 Slow |
| Data size | 📦 Only needed bands | 📦 Full products (~1GB) |
| Processing | 🔄 Immediate | 🔄 Manual extraction |
| Format | 📊 Cloud-optimized | 📊 Standard |

## How It Works

1. **STAC Search**: Queries public STAC catalog for Sentinel-2 products
2. **Direct Loading**: Uses `odc-stac` to load only required bands
3. **Scaling**: Applies scale/offset corrections automatically  
4. **Index Calculation**: Computes vegetation and water indices
5. **Export**: Saves as cloud-optimized GeoTIFF files

## Customization

### Custom Date Range
```python
# Specific date
search_results = downloader.search_items(..., start_date="2024-01-15", end_date="2024-01-15")

# Date range
search_results = downloader.search_items(..., start_date="2024-01-01", end_date="2024-01-31")
```

### Custom Filters
```python
search_results = downloader.search_items(
    ...,
    cloud_cover_max=10,  # Very low cloud cover
    # Add custom STAC query filters
    s2_vegetation_percentage={"gt": 25}  # High vegetation areas
)
```

### Custom Resolution
```python
data = downloader.load_data(search_results, resolution=20)  # 20m instead of 10m
```

## STAC Catalog Source

Uses Element 84's Earth Search STAC API: `https://earth-search.aws.element84.com/v1`

This provides access to Sentinel-2 Level-2A data hosted on AWS.

## Troubleshooting

### No products found
- Check your bounding box coordinates
- Increase cloud cover threshold
- Try a different date range
- Ensure coordinates are in correct format (lat/lon)

### Memory issues
- Reduce the area of interest
- Use higher resolution (lower detail)
- Process smaller date ranges

### Installation issues
```bash
# Install GDAL first on macOS
brew install gdal

# Install GDAL first on Ubuntu
sudo apt-get install gdal-bin libgdal-dev

# Then install Python packages
pip install --no-deps rasterio
pip install -r requirements.txt
```

## References

- [Medium Article: Download Sentinel Data within seconds](https://medium.com/rotten-grapes/download-sentinel-data-within-seconds-in-python-8cc9a8c3e23c)
- [STAC Specification](https://stacspec.org/)
- [pystac-client Documentation](https://pystac-client.readthedocs.io/)
- [odc-stac Documentation](https://odc-stac.readthedocs.io/) 