#!/usr/bin/env python3
"""
Example usage of the STAC-based Sentinel-2 data downloader script.
This file shows how to customize the download parameters for different regions.
"""

from query_sentinel_data import SentinelSTACDownloader

def download_for_region(name: str, north: float, south: float, east: float, west: float,
                       start_date: str = None, end_date: str = None, max_cloud_cover: float = 30.0):
    """
    Download Sentinel-2 data for a specific region using STAC.
    
    Args:
        name: Name of the region (for output directory)
        north, south, east, west: Bounding box coordinates
        start_date, end_date: Date range (YYYY-MM-DD format)
        max_cloud_cover: Maximum cloud coverage percentage
    """
    print(f"\n{'='*60}")
    print(f"Downloading Sentinel-2 data for: {name}")
    print(f"{'='*60}")
    
    try:
        # Initialize downloader
        downloader = SentinelSTACDownloader()
        
        # Search for items
        search_results = downloader.search_items(
            north=north, south=south, east=east, west=west,
            start_date=start_date, end_date=end_date,
            cloud_cover_max=max_cloud_cover
        )
        
        # Check if any items were found
        items_list = list(search_results.items())
        if not items_list:
            print(f"❌ No products found for {name}")
            return False
        
        # Create geometry and load data
        geometry = downloader.create_geometry(north, south, east, west)
        data = downloader.load_data(
            search_results=search_results,
            geometry=geometry,
            resolution=10
        )
        
        if data is None:
            print(f"❌ Failed to load data for {name}")
            return False
        
        # Apply scaling and calculate indices
        scaled_data = downloader.apply_scaling(data, search_results)
        final_data = downloader.calculate_indices(scaled_data)
        
        # Save data
        output_dir = f"sentinel_data_{name.lower().replace(' ', '_')}"
        saved_files = downloader.save_data(
            data=final_data,
            output_dir=output_dir,
            prefix=f"sentinel2_{name.lower().replace(' ', '_')}"
        )
        
        if saved_files:
            print(f"✅ Successfully downloaded data for {name}")
            print(f"   Output directory: {output_dir}")
            print(f"   Files created: {len(saved_files)}")
            return True
        else:
            print(f"❌ Failed to save data for {name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {name}: {e}")
        return False

def main():
    """
    Example usage with multiple regions using STAC-based approach.
    """
    print("STAC-based Sentinel-2 Data Downloader Examples")
    print("=" * 60)
    print("No credentials required - using public STAC catalogs!")
    print()
    
    # Define regions of interest with their bounding boxes
    regions = [
        {
            "name": "San Francisco Bay Area",
            "north": 37.9,
            "south": 37.3,
            "east": -122.0,
            "west": -122.8,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        },
        {
            "name": "Los Angeles",
            "north": 34.3,
            "south": 33.7,
            "east": -118.0,
            "west": -118.7,
            "start_date": "2024-01-01", 
            "end_date": "2024-01-31"
        },
        {
            "name": "New York City",
            "north": 40.9,
            "south": 40.5,
            "east": -73.7,
            "west": -74.3,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        },
        {
            "name": "London UK",
            "north": 51.6,
            "south": 51.4,
            "east": 0.2,
            "west": -0.4,
            "start_date": "2024-06-01",
            "end_date": "2024-06-30"
        }
    ]
    
    # Download data for each region
    successful_downloads = 0
    
    for region in regions:
        try:
            success = download_for_region(
                name=region["name"],
                north=region["north"],
                south=region["south"], 
                east=region["east"],
                west=region["west"],
                start_date=region["start_date"],
                end_date=region["end_date"],
                max_cloud_cover=25.0  # Slightly higher for better results
            )
            
            if success:
                successful_downloads += 1
                
        except Exception as e:
            print(f"Error processing {region['name']}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {successful_downloads}/{len(regions)} regions")
    print("\nTarget bands downloaded:")
    print("  blue (B02): Blue band (490nm) - 10m resolution")
    print("  green (B03): Green band (560nm) - 10m resolution") 
    print("  red (B04): Red band (665nm) - 10m resolution")
    print("  nir (B08): Near-Infrared band (842nm) - 10m resolution")
    print("  visual (TCI): True Color Image - RGB composite")
    print("\nCalculated indices:")
    print("  NDVI: Normalized Difference Vegetation Index")
    print("  NDWI: Normalized Difference Water Index")
    print("  EVI: Enhanced Vegetation Index")
    print("\nAdvantages of STAC approach:")
    print("  ✅ No authentication required")
    print("  ✅ Much faster than traditional downloads")
    print("  ✅ Direct access to specific bands")
    print("  ✅ Cloud-optimized data format")
    print("  ✅ Built-in data processing capabilities")

if __name__ == "__main__":
    main() 