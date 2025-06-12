import os
import json
import re
import glob
import numpy as np
from PIL import Image
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pyproj import Transformer
from display_kmz import (
    extract_kmz_content,
    extract_polygons_from_kml,
    get_item_name_from_filename,
    get_polygon_bounds,
    get_polygon_bounds_from_single_polygon
)

# --- KMZ Parsing Functions (from display_kmz.py) ---

def extract_kmz_content(kmz_file_path: str) -> str:
    """Extract KML content from a KMZ file."""
    with zipfile.ZipFile(kmz_file_path, 'r') as kmz:
        kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
        if not kml_files:
            raise ValueError("No KML files found in the KMZ archive")
        with kmz.open(kml_files[0]) as kml:
            return kml.read().decode('utf-8')

def extract_polygon_names_from_kml(kml_content: str) -> List[str]:
    """Extract only the polygon names from KML content."""
    root = ET.fromstring(kml_content)
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    names = []
    placemarks = root.findall('.//kml:Placemark', namespace)
    for placemark in placemarks:
        polygon_elem = placemark.find('.//kml:Polygon', namespace)
        if polygon_elem is not None:
            name_elem = placemark.find('kml:name', namespace)
            if name_elem is not None and name_elem.text:
                names.append(name_elem.text)
    return names

# --- Stitching Logic ---

def group_polygons(polygon_names: List[str]) -> Dict[str, List[str]]:
    """Groups polygon names based on a common prefix (e.g., 'TAP_A03_2012')."""
    groups = {}
    # This regex is designed to capture prefixes like 'TAP_A03_2012' or 'GO1_A01_2018'
    pattern = re.compile(r'^([A-Z0-9]+_[A-Z]{1,3}[0-9]{2}_[0-9]{4})')
    
    for name in polygon_names:
        match = pattern.match(name)
        if match:
            group_key = match.group(1)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(name)
        else:
            # Fallback for names that don't match, group them by the first part of the name
            fallback_key = name.split('_')[0]
            if fallback_key not in groups:
                groups[fallback_key] = []
            groups[fallback_key].append(name)
            
    return groups

def stitch_group_images(group_key: str, polygon_names: List[str], dtm_dir: str, output_dir: str, polygons: List[Dict[str, Any]]):
    """Stitches all DTM images for a given group of polygons."""
    print(f"\nStitching group: {group_key}...")
    
    # Get bounds from KMZ polygon data
    bounds = get_polygon_bounds(group_key, polygons)
    if bounds["max_lat"] == -180 or bounds["min_lat"] == 180 or bounds["max_lon"] == -180 or bounds["min_lon"] == 180:
        print(f"  - Warning: Invalid bounds for group {group_key}. Skipping.")
        return

    # Default resolution (in meters)
    resolution = 0.5

    # Calculate the size of the stitched image using the polygon bounds
    # Convert lat/lon to UTM coordinates for consistent pixel calculations
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
    
    # Transform bounds to UTM
    min_x, min_y = transformer_to_utm.transform(bounds["min_lon"], bounds["min_lat"])
    max_x, max_y = transformer_to_utm.transform(bounds["max_lon"], bounds["max_lat"])
    
    # Calculate dimensions
    stitched_width = int(np.ceil((max_x - min_x) / resolution))
    stitched_height = int(np.ceil((max_y - min_y) / resolution))
    
    print(f"  - Creating canvas of size: {stitched_width} x {stitched_height} pixels")

    # Check for reasonable dimensions
    if stitched_width > 100000 or stitched_height > 100000:
        print(f"  - Warning: Invalid stitched dimensions ({stitched_width} x {stitched_height}). Skipping group {group_key}.")
        return
    
    # Create a blank canvas with NaN support
    stitched_array = np.full((stitched_height, stitched_width), np.nan, dtype=np.float32)

    # Process each polygon in the group
    for name in polygon_names:
        # Find the corresponding polygon data
        polygon = next((p for p in polygons if p['name'] == name), None)
        if not polygon:
            print(f"  - Warning: No polygon data found for {name}. Skipping.")
            continue

        # Get the raw DTM data
        file_stem = os.path.splitext(name)[0]
        raw_data_path = os.path.join(dtm_dir, f"{file_stem}_dtm_csf.npy")
        if not os.path.exists(raw_data_path):
            print(f"  - Warning: Raw DTM data for {name} not found. Skipping.")
            continue
            
        dtm_array = np.load(raw_data_path)
        
        # Get the tile bounds from polygon coordinates
        tile_bounds = get_polygon_bounds_from_single_polygon(polygon)
        tile_min_x, tile_min_y = transformer_to_utm.transform(tile_bounds["min_lon"], tile_bounds["min_lat"])
        
        # Calculate pixel offsets on the main canvas
        x_offset = int(np.round((tile_min_x - min_x) / resolution))
        y_offset = int(np.round((tile_min_y - min_y) / resolution))
        
        h, w = dtm_array.shape
        
        # The y-axis needs to be flipped because array indices start from the top
        y_pos = stitched_height - (y_offset + h)
        if y_pos < 0: y_pos = 0
        
        # Combine the images, only overwriting NaN areas
        target_slice = stitched_array[y_pos:y_pos+h, x_offset:x_offset+w]
        if target_slice.shape != dtm_array.shape:
            print(f"  - Warning: Shape mismatch for {name}. Fitting DTM array to target slice.")
            min_h = min(target_slice.shape[0], dtm_array.shape[0])
            min_w = min(target_slice.shape[1], dtm_array.shape[1])
            fitted_dtm = np.zeros(target_slice.shape, dtype=dtm_array.dtype)
            fitted_dtm[:min_h, :min_w] = dtm_array[:min_h, :min_w]
            mask = np.isnan(target_slice)
            target_slice[mask] = fitted_dtm[mask]
        else:
            mask = np.isnan(target_slice) & ~np.isnan(dtm_array)
            target_slice[mask] = dtm_array[mask]

    # Normalize the array to 0-255 for saving as an image
    valid_pixels = stitched_array[~np.isnan(stitched_array)]
    if valid_pixels.size > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            normalized_array = (stitched_array - min_val) * (255.0 / (max_val - min_val))
            normalized_array[np.isnan(normalized_array)] = 0
        else:
            normalized_array = np.full(stitched_array.shape, 128)
            normalized_array[np.isnan(normalized_array)] = 0
    else:
        normalized_array = np.zeros(stitched_array.shape)

    final_image = Image.fromarray(normalized_array.astype(np.uint8), mode='L')
    
    output_path = os.path.join(output_dir, f"{group_key}_stitched.png")
    os.makedirs(output_dir, exist_ok=True)
    final_image.save(output_path)

    # Save location information using the polygon bounds
    info = {
        "group_key": group_key,
        "bounds_utm": {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y)
        },
        "bounds_latlon": {
            "min_lat": bounds["min_lat"],
            "max_lat": bounds["max_lat"],
            "min_lon": bounds["min_lon"],
            "max_lon": bounds["max_lon"]
        },
        "center_latlon": {
            "lat": (bounds["min_lat"] + bounds["max_lat"]) / 2,
            "lon": (bounds["min_lon"] + bounds["max_lon"]) / 2
        },
        "resolution": resolution
    }
    
    info_path = os.path.join(output_dir, f"{group_key}_stitched_location.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  -> Saved stitched image to {output_path}")
    print(f"  -> Saved location info to {info_path}")

def main():
    """Main function to run the stitching process."""
    # --- Configuration ---
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    dtm_images_dir = "exp/dtm_images_csf"
    stitched_output_dir = "stitched_images_v2"
    # ---------------------

    print("Starting DTM Stitching Process")
    print("=" * 40)

    if not os.path.exists(kmz_file_path):
        print(f"Error: KMZ file not found at '{kmz_file_path}'")
        return
        
    if not os.path.exists(dtm_images_dir):
        print(f"Error: DTM image directory not found at '{dtm_images_dir}'")
        print("Please run 'process_lidar.py' first to generate the DTMs and their metadata.")
        return

    # 1. Read polygon data from KMZ
    print("1. Parsing KMZ file to get polygon layout...")
    kml_content = extract_kmz_content(kmz_file_path)
    polygons = extract_polygons_from_kml(kml_content)
    print(f"   Found {len(polygons)} polygons in the KMZ file.")

    # 2. Group polygons
    print("\n2. Grouping contiguous polygons...")
    polygon_names = [polygon['name'] for polygon in polygons]
    groups = group_polygons(polygon_names)
    print(f"   Grouped into {len(groups)} contiguous sets.")
    for key, items in groups.items():
        print(f"   - Group '{key}': {len(items)} tiles")

    # 3. Stitch images for each group
    print("\n3. Stitching images for each group...")
    for group_key, names in groups.items():
        stitch_group_images(
            group_key=group_key,
            polygon_names=names,
            dtm_dir=dtm_images_dir,
            output_dir=stitched_output_dir,
            polygons=polygons
        )
        
    print("\n" + "=" * 40)
    print("Stitching process complete.")
    print(f"All stitched images are saved in '{os.path.abspath(stitched_output_dir)}'.")

if __name__ == "__main__":
    main()