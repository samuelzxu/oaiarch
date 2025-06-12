import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

try:
    from pystac_client import Client
    from odc.stac import load
    import odc.geo
    import rasterio
    import xarray as xr
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Install with: pip install pystac-client odc-stac rasterio xarray matplotlib")
    exit(1)

class SentinelSTACDownloader:
    """
    A class to download Sentinel-2 L2A data using STAC catalogs for specified 
    spectral bands within a given lat/lon bounding box.
    
    This approach is much faster than traditional methods as it uses STAC
    (SpatioTemporal Asset Catalog) to directly access and load only the
    required bands without downloading entire products.
    """
    
    def __init__(self, stac_url: str = "https://earth-search.aws.element84.com/v1"):
        """
        Initialize the STAC-based Sentinel data downloader.
        
        Args:
            stac_url: URL of the STAC catalog endpoint
        """
        self.client = Client.open(stac_url)
        self.collection = "sentinel-2-l2a"
        
        # Target bands (10m resolution)
        self.target_bands = ['red', 'green', 'blue', 'nir', 'visual']  # B04, B03, B02, B08, TCI
        
        print(f"Connected to STAC catalog: {stac_url}")
        print(f"Target collection: {self.collection}")
        print(f"Target bands: {self.target_bands}")
    
    def create_bbox(self, north: float, south: float, east: float, west: float) -> list:
        """
        Create a bounding box list from coordinates.
        
        Args:
            north: Northern latitude boundary
            south: Southern latitude boundary
            east: Eastern longitude boundary
            west: Western longitude boundary
            
        Returns:
            Bounding box as [west, south, east, north]
        """
        return [west, south, east, north]
    
    def create_geometry(self, north: float, south: float, east: float, west: float) -> dict:
        """
        Create a GeoJSON polygon from bounding box coordinates.
        
        Args:
            north: Northern latitude boundary
            south: Southern latitude boundary
            east: Eastern longitude boundary
            west: Western longitude boundary
            
        Returns:
            GeoJSON polygon dictionary
        """
        geometry = {
            "coordinates": [
                [
                    [west, south],
                    [east, south], 
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ],
            "type": "Polygon",
        }
        return geometry
    
    def search_items(self, 
                    north: float,
                    south: float, 
                    east: float,
                    west: float,
                    start_date: str = None,
                    end_date: str = None,
                    cloud_cover_max: float = 30.0,
                    **kwargs) -> object:
        """
        Search for Sentinel-2 L2A items within the specified parameters.
        
        Args:
            north: Northern latitude boundary
            south: Southern latitude boundary
            east: Eastern longitude boundary
            west: Western longitude boundary
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            cloud_cover_max: Maximum cloud coverage percentage
            **kwargs: Additional query parameters
            
        Returns:
            STAC search object
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
        # Create geometry
        geometry = self.create_geometry(north, south, east, west)
        
        # Create date range string
        if start_date == end_date:
            datetime_str = start_date
        else:
            datetime_str = f"{start_date}/{end_date}"
        
        print(f"Searching for Sentinel-2 L2A products...")
        print(f"  Area: {west}°W to {east}°E, {south}°S to {north}°N")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Max cloud cover: {cloud_cover_max}%")
        
        # Build query filters
        query_filters = {
            "eo:cloud_cover": {"lt": cloud_cover_max}
        }
        
        # Add any additional query parameters
        query_filters.update(kwargs)
        
        # Search for items
        search = self.client.search(
            collections=[self.collection],
            intersects=geometry,
            datetime=datetime_str,
            query=query_filters
        )
        
        # Get items as list to count them
        items = list(search.items())
        print(f"Found {len(items)} products matching criteria.")
        
        return search
    
    def load_data(self, 
                 search_results: object,
                 geometry: dict = None,
                 resolution: int = 10,
                 **load_kwargs) -> xr.Dataset:
        """
        Load the satellite data from search results into an xarray Dataset.
        
        Args:
            search_results: STAC search results object
            geometry: GeoJSON geometry to clip data (optional)
            resolution: Target resolution in meters
            **load_kwargs: Additional parameters for odc.stac.load
            
        Returns:
            xarray Dataset with loaded bands
        """
        print(f"Loading data with {resolution}m resolution...")
        
        # Default load parameters
        default_params = {
            "bands": self.target_bands,
            "resolution": resolution,
            "groupby": "solar_day",
            "chunks": {"x": 2048, "y": 2048},
        }
        
        # Add geometry if provided
        if geometry:
            default_params["geopolygon"] = geometry
        
        # Update with user parameters
        default_params.update(load_kwargs)
        
        # Load the data
        try:
            data = load(search_results.items(), **default_params)
            
            if len(data.time) == 0:
                print("Warning: No data loaded. Check your search parameters.")
                return None
                
            print(f"Successfully loaded data:")
            print(f"  Time steps: {len(data.time)}")
            print(f"  Bands: {list(data.data_vars.keys())}")
            print(f"  Spatial dimensions: {data.sizes}")
            print(f"  CRS: {data.odc.crs}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def apply_scaling(self, data: xr.Dataset, search_results: object) -> xr.Dataset:
        """
        Apply scale and offset corrections to the data.
        
        Args:
            data: xarray Dataset with raw satellite data
            search_results: STAC search results to get scaling info
            
        Returns:
            xarray Dataset with scaled data
        """
        print("Applying scale and offset corrections...")
        
        try:
            # Get the first item to extract scaling information
            first_item = next(search_results.items())
            
            scaled_data = data.copy()
            
            # Apply scaling for each band
            for band in self.target_bands:
                if band in data.data_vars:
                    try:
                        # Get scale and offset from STAC item
                        asset_info = first_item.assets.get(band, {})
                        raster_bands = asset_info.extra_fields.get('raster:bands', [{}])
                        
                        if raster_bands:
                            scale = raster_bands[0].get('scale', 1.0)
                            offset = raster_bands[0].get('offset', 0.0)
                            
                            # Apply scaling
                            scaled_data[band] = data[band] * scale + offset
                            print(f"  Applied scaling to {band}: scale={scale}, offset={offset}")
                        else:
                            print(f"  No scaling info found for {band}, using raw values")
                            
                    except Exception as e:
                        print(f"  Warning: Could not apply scaling to {band}: {e}")
            
            return scaled_data
            
        except Exception as e:
            print(f"Warning: Could not apply scaling: {e}")
            return data
    
    def calculate_indices(self, data: xr.Dataset) -> xr.Dataset:
        """
        Calculate vegetation indices from the loaded data.
        
        Args:
            data: xarray Dataset with satellite bands
            
        Returns:
            xarray Dataset with added indices
        """
        print("Calculating vegetation indices...")
        
        indices_data = data.copy()
        
        try:
            # Calculate NDVI if red and nir bands are available
            if 'red' in data.data_vars and 'nir' in data.data_vars:
                indices_data['ndvi'] = (data.nir - data.red) / (data.nir + data.red)
                print("  Added NDVI (Normalized Difference Vegetation Index)")
            
            # Calculate NDWI if green and nir bands are available  
            if 'green' in data.data_vars and 'nir' in data.data_vars:
                indices_data['ndwi'] = (data.green - data.nir) / (data.green + data.nir)
                print("  Added NDWI (Normalized Difference Water Index)")
                
            # Calculate EVI if blue, red, and nir bands are available
            if all(band in data.data_vars for band in ['blue', 'red', 'nir']):
                indices_data['evi'] = 2.5 * ((data.nir - data.red) / 
                                           (data.nir + 6 * data.red - 7.5 * data.blue + 1))
                print("  Added EVI (Enhanced Vegetation Index)")
        
        except Exception as e:
            print(f"Warning: Error calculating indices: {e}")
            
        return indices_data
    
    def _normalize_for_png(self, data_array: xr.DataArray, percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize data for PNG saving by clipping outliers and scaling to 0-255.
        
        Args:
            data_array: xarray DataArray to normalize
            percentile_clip: Tuple of (low_percentile, high_percentile) for clipping outliers
            
        Returns:
            Normalized numpy array scaled to 0-255 range
        """
        # Convert to numpy array and handle NaN values
        data = data_array.values
        
        # For indices like NDVI, NDWI which can have negative values, handle specially
        var_name = data_array.name if hasattr(data_array, 'name') else 'unknown'
        
        if var_name in ['ndvi', 'ndwi', 'evi']:
            # For vegetation indices, use a fixed range
            if var_name == 'ndvi':
                vmin, vmax = -1, 1
            elif var_name == 'ndwi':
                vmin, vmax = -1, 1
            elif var_name == 'evi':
                vmin, vmax = -1, 2
        else:
            # For other bands, use percentile clipping
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return np.zeros_like(data, dtype=np.uint8)
            
            vmin, vmax = np.percentile(valid_data, percentile_clip)
        
        # Clip and normalize to 0-255
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        
        # Handle NaN values by setting them to 0
        normalized[np.isnan(data)] = 0
        
        return normalized
    
    def save_data(self, 
                 data: xr.Dataset, 
                 output_dir: str, 
                 prefix: str = "sentinel2_data") -> Dict[str, str]:
        """
        Save the loaded data to various formats (TIFF and PNG).
        
        Args:
            data: xarray Dataset to save
            output_dir: Directory to save files
            prefix: Prefix for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        print(f"Saving data to {output_dir}...")
        
        # Save each band and index as both GeoTIFF and PNG
        for i, time_step in enumerate(reversed(data.time)):
            for var_name in data.data_vars:
                time_str = pd.to_datetime(time_step.values).strftime('%Y%m%d')
                
                # Select data for this time step
                var_data = data[var_name].sel(time=time_step)
                
                # Save as GeoTIFF using odc.geo
                tif_filename = f"{prefix}_{var_name}_{time_str}.tif"
                tif_filepath = output_path / tif_filename
                odc.geo.xr.write_cog(
                    var_data, 
                    fname=str(tif_filepath), 
                    overwrite=True
                )
                saved_files[f"{var_name}_{time_str}_tif"] = str(tif_filepath)
                print(f"  Saved {tif_filename}")
                
                # Save as PNG
                png_filename = f"{prefix}_{var_name}_{time_str}.png"
                png_filepath = output_path / png_filename
                
                # Normalize data for PNG
                normalized_data = self._normalize_for_png(var_data)
                
                # Save PNG using matplotlib
                plt.figure(figsize=(10, 10))
                plt.imshow(normalized_data, cmap='gray' if var_name not in ['visual'] else None)
                plt.axis('off')
                plt.title(f"{var_name.upper()} - {time_str}", fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(str(png_filepath), dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
                saved_files[f"{var_name}_{time_str}_png"] = str(png_filepath)
                print(f"  Saved {png_filename}")
            break
        
        # Save metadata
        metadata = {
            'bands': list(data.data_vars.keys()),
            'time_steps': [pd.to_datetime(t.values).isoformat() for t in data.time],
            'crs': str(data.odc.crs),
            'spatial_dimensions': dict(data.sizes),
            'bounds': {
                'left': float(data.x.min()),
                'right': float(data.x.max()), 
                'bottom': float(data.y.min()),
                'top': float(data.y.max())
            },
            'processing_date': datetime.now().isoformat(),
            'saved_files': saved_files
        }
        
        metadata_path = output_path / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path.name}")
        
        return saved_files

def main():
    """
    Main function to demonstrate usage of the STAC-based Sentinel data downloader.
    """
    # --- Configuration ---
    # Bounding box coordinates (example: San Francisco Bay Area)
    NORTH = 37.9   # Northern latitude
    SOUTH = 37.3   # Southern latitude
    EAST = -122.0  # Eastern longitude  
    WEST = -122.8  # Western longitude
    
    # Search parameters
    START_DATE = "2024-01-01"   # Format: YYYY-MM-DD
    END_DATE = "2024-01-31"     # Format: YYYY-MM-DD
    MAX_CLOUD_COVER = 20        # Maximum cloud coverage percentage
    RESOLUTION = 10             # Spatial resolution in meters
    
    # Output directory
    OUTPUT_DIR = "sentinel_stac_data"
    # ---------------------
    
    print("Sentinel-2 L2A Data Downloader (STAC Method)")
    print("=" * 50)
    print(f"Target bands: B02 (blue), B03 (green), B04 (red), B08 (nir), TCI (visual)")
    print(f"Bounding box: {WEST}° to {EAST}°, {SOUTH}° to {NORTH}°")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"Resolution: {RESOLUTION}m")
    print()
    
    try:
        # Initialize downloader
        downloader = SentinelSTACDownloader()
        
        # Search for items
        search_results = downloader.search_items(
            north=NORTH,
            south=SOUTH,
            east=EAST, 
            west=WEST,
            start_date=START_DATE,
            end_date=END_DATE,
            cloud_cover_max=MAX_CLOUD_COVER
        )
        
        # Check if any items were found
        items_list = list(search_results.items())
        if not items_list:
            print("No products found matching the criteria.")
            return
        
        # Display found products
        print("\nFound products:")
        for i, item in enumerate(items_list[:5]):  # Show first 5
            properties = item.properties
            print(f"  {i+1}. {item.id}")
            print(f"     Date: {properties.get('datetime', 'N/A')}")
            print(f"     Cloud cover: {properties.get('eo:cloud_cover', 'N/A')}%")
            if i >= 4 and len(items_list) > 5:
                print(f"     ... and {len(items_list) - 5} more products")
                break
        print()
        
        # Create geometry for data loading
        geometry = downloader.create_geometry(NORTH, SOUTH, EAST, WEST)
        
        # Load the data
        data = downloader.load_data(
            search_results=search_results,
            geometry=geometry,
            resolution=RESOLUTION
        )
        
        if data is None:
            print("Failed to load any data.")
            return
        
        # Apply scaling corrections
        scaled_data = downloader.apply_scaling(data, search_results)
        
        # Calculate vegetation indices
        final_data = downloader.calculate_indices(scaled_data)
        
        # Save the data
        saved_files = downloader.save_data(
            data=final_data,
            output_dir=OUTPUT_DIR,
            prefix="sentinel2_l2a"
        )
        
        print(f"\n{'='*50}")
        print("Download and processing complete!")
        print(f"Files saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Total files created: {len(saved_files)}")
        print("\nOutput Formats:")
        print("  TIFF: Georeferenced raster files for GIS analysis")
        print("  PNG: Visual preview images for quick inspection")
        print("\nBand Information:")
        print("  blue (B02): Blue band (490nm) - 10m resolution")
        print("  green (B03): Green band (560nm) - 10m resolution")
        print("  red (B04): Red band (665nm) - 10m resolution") 
        print("  nir (B08): Near-Infrared band (842nm) - 10m resolution")
        print("  visual (TCI): True Color Image - RGB composite")
        print("\nCalculated Indices:")
        print("  NDVI: Normalized Difference Vegetation Index")
        print("  NDWI: Normalized Difference Water Index") 
        print("  EVI: Enhanced Vegetation Index")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 