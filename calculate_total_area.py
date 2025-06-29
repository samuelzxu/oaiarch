"""
Calculate the total area in hectares covered by LiDAR files from the KMZ inventory.

This script uses the KMZ file to extract polygon coordinates and calculates 
the total area covered by all LiDAR tiles in hectares.

Dependencies
------------
pip install pyproj geopy
"""

import math
from typing import List, Dict, Any
from display_kmz import extract_kmz_content, extract_polygons_from_kml
from geopy.distance import geodesic


def calculate_polygon_area_geodesic(coordinates: List[tuple]) -> float:
    """
    Calculate the area of a polygon using geodesic calculations.
    
    Args:
        coordinates: List of (lat, lon) tuples representing polygon vertices
        
    Returns:
        Area in square meters
    """
    if len(coordinates) < 3:
        return 0.0
    
    # Ensure the polygon is closed
    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + [coordinates[0]]
    
    # Use the Shoelace formula adapted for geodesic calculations
    total_area = 0.0
    n = len(coordinates)
    
    for i in range(n - 1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i + 1]
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        # Calculate the area contribution of this edge using spherical excess
        dlon = lon2_rad - lon1_rad
        
        # Simplified spherical area calculation for small polygons
        # This is an approximation but should be sufficient for our purposes
        area_contribution = dlon * (2 + math.sin(lat1_rad) + math.sin(lat2_rad))
        total_area += area_contribution
    
    # Earth's radius in meters
    earth_radius = 6371000.0
    
    # Convert to square meters
    area_sq_meters = abs(total_area) * earth_radius * earth_radius / 2.0
    
    return area_sq_meters


def calculate_polygon_area_simple(coordinates: List[tuple]) -> float:
    """
    Calculate the area of a polygon using a simpler approximation method.
    
    This method treats lat/lon as if they were on a flat surface with appropriate
    scaling for latitude. It's less accurate than geodesic but faster and simpler.
    
    Args:
        coordinates: List of (lat, lon) tuples representing polygon vertices
        
    Returns:
        Area in square meters
    """
    if len(coordinates) < 3:
        return 0.0
    
    # Calculate the average latitude for scaling
    avg_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    lat_scale = math.cos(math.radians(avg_lat))
    
    # Convert degrees to meters (approximately)
    # 1 degree of latitude ≈ 111,319 meters
    # 1 degree of longitude ≈ 111,319 * cos(latitude) meters
    lat_to_meters = 111319.0
    lon_to_meters = 111319.0 * lat_scale
    
    # Apply Shoelace formula
    n = len(coordinates)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        lat_i, lon_i = coordinates[i]
        lat_j, lon_j = coordinates[j]
        
        # Convert to meters
        x_i = lon_i * lon_to_meters
        y_i = lat_i * lat_to_meters
        x_j = lon_j * lon_to_meters
        y_j = lat_j * lat_to_meters
        
        area += x_i * y_j - x_j * y_i
    
    return abs(area) / 2.0


def calculate_total_lidar_area(kmz_file_path: str = "cms_brazil_lidar_tile_inventory.kmz") -> Dict[str, float]:
    """
    Calculate the total area covered by all LiDAR files in the KMZ inventory.
    
    Args:
        kmz_file_path: Path to the KMZ file containing polygon data
        
    Returns:
        Dictionary with area statistics in different units
    """
    print("Extracting KML content from KMZ file...")
    kml_content = extract_kmz_content(kmz_file_path)
    
    print("Parsing polygons from KML...")
    polygons = extract_polygons_from_kml(kml_content)
    
    print(f"Found {len(polygons)} polygons")
    
    total_area_sq_meters = 0.0
    individual_areas = []
    
    print("\nCalculating areas for each polygon...")
    
    for i, polygon in enumerate(polygons):
        coordinates = polygon['coordinates']
        name = polygon['name']
        
        # Calculate area using the simple method (you can switch to geodesic if needed)
        area_sq_meters = calculate_polygon_area_simple(coordinates)
        individual_areas.append({
            'name': name,
            'area_sq_meters': area_sq_meters,
            'area_hectares': area_sq_meters / 10000.0
        })
        
        total_area_sq_meters += area_sq_meters
        
        if (i + 1) % 100 == 0:  # Progress indicator
            print(f"  Processed {i + 1}/{len(polygons)} polygons...")
    
    # Convert to different units
    total_area_hectares = total_area_sq_meters / 10000.0  # 1 hectare = 10,000 m²
    total_area_sq_km = total_area_sq_meters / 1000000.0   # 1 km² = 1,000,000 m²
    
    # Calculate statistics
    areas_hectares = [poly['area_hectares'] for poly in individual_areas]
    avg_area_hectares = sum(areas_hectares) / len(areas_hectares)
    min_area_hectares = min(areas_hectares)
    max_area_hectares = max(areas_hectares)
    
    results = {
        'total_area_sq_meters': total_area_sq_meters,
        'total_area_hectares': total_area_hectares,
        'total_area_sq_km': total_area_sq_km,
        'num_polygons': len(polygons),
        'avg_area_hectares': avg_area_hectares,
        'min_area_hectares': min_area_hectares,
        'max_area_hectares': max_area_hectares,
        'individual_areas': individual_areas
    }
    
    return results


def print_area_summary(results: Dict[str, float]) -> None:
    """
    Print a formatted summary of the area calculations.
    
    Args:
        results: Dictionary returned by calculate_total_lidar_area()
    """
    print("\n" + "="*60)
    print("LIDAR COVERAGE AREA SUMMARY")
    print("="*60)
    
    print(f"Total number of LiDAR tiles: {results['num_polygons']:,}")
    print(f"Total area covered:")
    print(f"  • {results['total_area_hectares']:,.2f} hectares")
    print(f"  • {results['total_area_sq_km']:,.2f} square kilometers")
    print(f"  • {results['total_area_sq_meters']:,.0f} square meters")
    
    print(f"\nIndividual tile statistics:")
    print(f"  • Average tile area: {results['avg_area_hectares']:.2f} hectares")
    print(f"  • Smallest tile: {results['min_area_hectares']:.2f} hectares")
    print(f"  • Largest tile: {results['max_area_hectares']:.2f} hectares")
    
    # Add some reference comparisons
    print(f"\nFor reference:")
    print(f"  • Total area is equivalent to {results['total_area_hectares']/100:.1f} square kilometers")
    print(f"  • That's about {results['total_area_hectares']/259:.1f} square miles")
    
    # Compare to well-known areas
    if results['total_area_sq_km'] > 1000:
        print(f"  • This is larger than some small countries!")
    
    print("="*60)


def save_detailed_results(results: Dict[str, float], output_file: str = "lidar_area_analysis.txt") -> None:
    """
    Save detailed results to a text file.
    
    Args:
        results: Dictionary returned by calculate_total_lidar_area()
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("LiDAR Coverage Area Analysis\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total number of LiDAR tiles: {results['num_polygons']:,}\n")
        f.write(f"Total area covered: {results['total_area_hectares']:,.2f} hectares\n")
        f.write(f"Total area covered: {results['total_area_sq_km']:,.2f} square kilometers\n")
        f.write(f"Total area covered: {results['total_area_sq_meters']:,.0f} square meters\n\n")
        
        f.write("Individual Tile Areas:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Tile Name':<40} {'Area (hectares)':<15}\n")
        f.write("-" * 50 + "\n")
        
        # Sort by area for easier reading
        sorted_areas = sorted(results['individual_areas'], 
                            key=lambda x: x['area_hectares'], reverse=True)
        
        for poly in sorted_areas:
            f.write(f"{poly['name']:<40} {poly['area_hectares']:<15.2f}\n")
    
    print(f"Detailed results saved to {output_file}")


def main():
    """Main function to run the area calculation."""
    try:
        print("Starting LiDAR area calculation...")
        
        # Calculate the total area
        results = calculate_total_lidar_area()
        
        # Print summary
        print_area_summary(results)
        
        # Save detailed results
        save_detailed_results(results)
        
        print(f"\nCalculation completed successfully!")
        print(f"TOTAL AREA: {results['total_area_hectares']:,.2f} hectares")
        
    except FileNotFoundError:
        print("Error: Could not find the KMZ file 'cms_brazil_lidar_tile_inventory.kmz'")
        print("Please make sure the file is in the current directory.")
    except Exception as e:
        print(f"Error during calculation: {str(e)}")


if __name__ == "__main__":
    main() 