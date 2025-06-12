import zipfile
import xml.etree.ElementTree as ET
import folium
from folium import plugins
import re
from typing import List, Tuple, Dict, Any
import os

def extract_kmz_content(kmz_file_path: str) -> str:
    """
    Extract KML content from KMZ file.
    KMZ files are just ZIP archives containing KML files.
    """
    with zipfile.ZipFile(kmz_file_path, 'r') as kmz:
        # Look for KML files in the archive
        kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
        
        if not kml_files:
            raise ValueError("No KML files found in the KMZ archive")
        
        # Use the first KML file found (usually 'doc.kml')
        kml_file = kml_files[0]
        
        with kmz.open(kml_file) as kml:
            return kml.read().decode('utf-8')

def parse_coordinates(coord_string: str) -> List[Tuple[float, float]]:
    """
    Parse coordinate string from KML format.
    KML coordinates are in format: longitude,latitude,altitude (space or newline separated)
    """
    coordinates = []
    
    # Clean up the coordinate string
    coord_string = coord_string.strip()
    
    # Split by whitespace and process each coordinate pair
    coord_pairs = re.split(r'\s+', coord_string)
    
    for pair in coord_pairs:
        if pair.strip():
            parts = pair.split(',')
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    coordinates.append((lat, lon))  # Folium expects (lat, lon)
                except ValueError:
                    continue
    
    return coordinates

def extract_polygons_from_kml(kml_content: str) -> List[Dict[str, Any]]:
    """
    Extract polygon data from KML content.
    """
    # Parse XML with namespace handling
    root = ET.fromstring(kml_content)
    
    # Define KML namespace
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    polygons = []
    
    # Find all Placemark elements that contain Polygon data
    placemarks = root.findall('.//kml:Placemark', namespace)
    
    for placemark in placemarks:
        # Get placemark name
        name_elem = placemark.find('kml:name', namespace)
        name = name_elem.text if name_elem is not None else "Unnamed Polygon"
        
        # Get description if available
        desc_elem = placemark.find('kml:description', namespace)
        description = desc_elem.text if desc_elem is not None else ""
        
        # Find polygon elements
        polygon_elem = placemark.find('.//kml:Polygon', namespace)
        
        if polygon_elem is not None:
            # Get outer boundary coordinates
            outer_boundary = polygon_elem.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
            
            if outer_boundary is not None:
                coordinates = parse_coordinates(outer_boundary.text)
                
                if coordinates:
                    polygon_data = {
                        'name': name,
                        'description': description,
                        'coordinates': coordinates,
                        'type': 'polygon'
                    }
                    
                    # Check for inner boundaries (holes)
                    inner_boundaries = polygon_elem.findall('.//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates', namespace)
                    if inner_boundaries:
                        holes = []
                        for inner in inner_boundaries:
                            hole_coords = parse_coordinates(inner.text)
                            if hole_coords:
                                holes.append(hole_coords)
                        polygon_data['holes'] = holes
                    
                    polygons.append(polygon_data)
    
    return polygons

def create_map_with_polygons(polygons: List[Dict[str, Any]], output_file: str = 'kmz_polygons_map.html'):
    """
    Create a Folium map with the extracted polygons.
    """
    if not polygons:
        print("No polygons found to display")
        return None
    
    # Calculate center point from all polygons
    all_lats = []
    all_lons = []
    
    for polygon in polygons:
        for lat, lon in polygon['coordinates']:
            all_lats.append(lat)
            all_lons.append(lon)
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add polygons to map
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    for i, polygon in enumerate(polygons):
        color = colors[i % len(colors)]
        
        # Create popup content
        popup_content = f"""
        <b>{polygon['name']}</b><br>
        {polygon['description'][:200]}{'...' if len(polygon['description']) > 200 else ''}
        """
        
        # Add polygon with holes if they exist
        locations = polygon['coordinates']
        holes = polygon.get('holes', [])
        
        folium.Polygon(
            locations=locations,
            holes=holes,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=polygon['name'],
            color=color,
            weight=2,
            fillColor=color,
            fillOpacity=0.3
        ).add_to(m)
    
    # Add a marker at the center
    folium.Marker(
        [center_lat, center_lon],
        popup="Map Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Map saved as {output_file}")
    
    return m


def get_item_name_from_filename(filename):
    num_underscores = filename.count('_')
    split_at = max(min(3,num_underscores),0)
    return '_'.join(filename.split('_')[:split_at])

def get_polygon_bounds(polygon_name, polygons):
    max_lat = -180
    max_lon = -180
    min_lat = 180
    min_lon = 180
    for poly in polygons:
        p_id = get_item_name_from_filename(poly["name"])
        if p_id == polygon_name:
            max_lat = max(max_lat, max([coord[0] for coord in poly['coordinates']]))
            max_lon = max(max_lon, max([coord[1] for coord in poly['coordinates']]))
            min_lat = min(min_lat, min([coord[0] for coord in poly['coordinates']]))
            min_lon = min(min_lon, min([coord[1] for coord in poly['coordinates']]))
    return {
        "max_lat": max_lat,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "min_lon": min_lon
    }

def get_polygon_bounds_from_single_polygon(polygon):
    """
    Get the bounds of a single polygon.
    """
    max_lat = max([coord[0] for coord in polygon['coordinates']])
    max_lon = max([coord[1] for coord in polygon['coordinates']])
    min_lat = min([coord[0] for coord in polygon['coordinates']])
    min_lon = min([coord[1] for coord in polygon['coordinates']])
    
    return {
        "max_lat": max_lat,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "min_lon": min_lon
    }

def extract_coordinates():
    """
    Main function to demonstrate the KMZ reading and mapping functionality.
    """
    # Replace with your KMZ file path
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    
    try:
        print("Extracting KML content from KMZ file...")
        kml_content = extract_kmz_content(kmz_file_path)
        
        print("Parsing polygons from KML...")
        polygons = extract_polygons_from_kml(kml_content)
        
        print(f"Found {len(polygons)} polygons")
        
        # Print summary of found polygons
        for i, polygon in enumerate(polygons):
            print(f"Polygon {i+1}: {polygon['name']} ({polygon['coordinates']})")
        
        all_polygon_names = set([get_item_name_from_filename(polygon['name']) for polygon in polygons])
        
        # print("Creating map...")
        stitched_dir = 'stitched_images'
        stitched_items = list(map(get_item_name_from_filename,filter(lambda x: '.png' in x, os.listdir(stitched_dir))))
        polygon_bounds = {}
        for item in stitched_items:
            
            if not(item in all_polygon_names):
                print('Item found not in polygons: ',item)
            else:
                polygon_bounds[item] = get_polygon_bounds(item, polygons)
        return polygon_bounds
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{kmz_file_path}'")
        print("Please update the kmz_file_path variable with the correct path to your KMZ file.")
    except Exception as e:
        print(f"Error processing KMZ file: {str(e)}")

if __name__ == "__main__":
    extract_coordinates()