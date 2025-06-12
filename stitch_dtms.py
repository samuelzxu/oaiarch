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
    
    # Collect metadata for all tiles in the group
    group_metadata = []
    for name in polygon_names:
        file_stem = os.path.splitext(name)[0]
        metadata_path = os.path.join(dtm_dir, f"{file_stem}_dtm_csf.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata['polygon_name'] = name
                group_metadata.append(metadata)
        else:
            print(f"  - Warning: Metadata for {name} not found. Skipping.")

    if not group_metadata:
        print(f"No DTM images found for group {group_key}. Skipping.")
        return

    # Determine the overall bounding box for the entire group using UTM coordinates from metadata
    global_min_x = min(m['bounds']['min_x'] for m in group_metadata)
    global_max_x = max(m['bounds']['max_x'] for m in group_metadata)
    global_min_y = min(m['bounds']['min_y'] for m in group_metadata)
    global_max_y = max(m['bounds']['max_y'] for m in group_metadata)
    
    # Use resolution from metadata
    resolution = group_metadata[0]['resolution']

    # Calculate the size of the stitched image
    stitched_width = int(np.ceil((global_max_x - global_min_x) / resolution))
    stitched_height = int(np.ceil((global_max_y - global_min_y) / resolution))
    
    print(f"  - Creating canvas of size: {stitched_width} x {stitched_height} pixels")
    print(f"  - UTM bounds: X({global_min_x:.2f}, {global_max_x:.2f}), Y({global_min_y:.2f}, {global_max_y:.2f})")

    # Check for reasonable dimensions
    if stitched_width > 100000 or stitched_height > 100000:
        print(f"  - Warning: Invalid stitched dimensions ({stitched_width} x {stitched_height}). Skipping group {group_key}.")
        return
    
    # Create a blank canvas with NaN support
    stitched_array = np.full((stitched_height, stitched_width), np.nan, dtype=np.float32)

    # Process each tile
    for metadata in group_metadata:
        raw_data_path = metadata.get('dtm_raw_data')
        if not raw_data_path or not os.path.exists(raw_data_path):
            print(f"  - Warning: Raw DTM data for {metadata['polygon_name']} not found. Skipping.")
            continue
            
        dtm_array = np.load(raw_data_path)
        
        # Get tile bounds from metadata (these are in UTM coordinates)
        tile_bounds = metadata['bounds']
        tile_min_x = tile_bounds['min_x']
        tile_min_y = tile_bounds['min_y']
        
        # Calculate pixel offsets on the main canvas
        x_offset = int(np.round((tile_min_x - global_min_x) / resolution))
        y_offset = int(np.round((tile_min_y - global_min_y) / resolution))
        
        h, w = dtm_array.shape
        
        # Calculate position in the stitched array
        # Note: DTM arrays are created with origin='lower' in matplotlib, so we need to handle this
        y_pos = stitched_height - (y_offset + h)
        if y_pos < 0: 
            print(f"  - Warning: Y position {y_pos} is negative for {metadata['polygon_name']}. Adjusting to 0.")
            y_pos = 0
        
        # Ensure we don't go out of bounds
        end_y = min(y_pos + h, stitched_height)
        end_x = min(x_offset + w, stitched_width)
        
        # Adjust the DTM array size if needed
        actual_h = end_y - y_pos
        actual_w = end_x - x_offset
        
        if actual_h <= 0 or actual_w <= 0:
            print(f"  - Warning: Invalid dimensions for {metadata['polygon_name']}. Skipping.")
            continue
            
        # Get the slice of the DTM array that fits
        dtm_slice = dtm_array[:actual_h, :actual_w]
        
        # Get the target slice in the stitched array
        target_slice = stitched_array[y_pos:end_y, x_offset:end_x]
        
        # Only paste where the canvas is currently empty (NaN)
        mask = np.isnan(target_slice) & ~np.isnan(dtm_slice)
        target_slice[mask] = dtm_slice[mask]
        
        print(f"  - Placed {metadata['polygon_name']} at position ({x_offset}, {y_pos}) with size ({actual_w}, {actual_h})")

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
        print(f"  - Warning: No valid pixels found for group {group_key}")
        normalized_array = np.zeros(stitched_array.shape)

    final_image = Image.fromarray(normalized_array.astype(np.uint8), mode='L')
    
    output_path = os.path.join(output_dir, f"{group_key}_stitched.png")
    os.makedirs(output_dir, exist_ok=True)
    final_image.save(output_path)

    # Convert UTM bounds to lat/lon for the location info
    transformer_to_latlon = Transformer.from_crs("EPSG:32721", "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer_to_latlon.transform(global_min_x, global_min_y)
    max_lon, max_lat = transformer_to_latlon.transform(global_max_x, global_max_y)

    # Save location information
    info = {
        "group_key": group_key,
        "bounds_utm": {
            "min_x": float(global_min_x),
            "max_x": float(global_max_x),
            "min_y": float(global_min_y),
            "max_y": float(global_max_y)
        },
        "bounds_latlon": {
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
            "min_lon": float(min_lon),
            "max_lon": float(max_lon)
        },
        "center_latlon": {
            "lat": float((min_lat + max_lat) / 2),
            "lon": float((min_lon + max_lon) / 2)
        },
        "resolution": resolution,
        "image_size": {
            "width": stitched_width,
            "height": stitched_height
        }
    }
    
    info_path = os.path.join(output_dir, f"{group_key}_stitched_location.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  -> Saved stitched image to {output_path}")
    print(f"  -> Saved location info to {info_path}")
    print(f"  -> Final image size: {stitched_width} x {stitched_height} pixels")
    print(f"  -> Valid pixels: {valid_pixels.size} / {stitched_array.size} ({100*valid_pixels.size/stitched_array.size:.1f}%)")

def main():
    """Main function to run the stitching process."""
    # --- Configuration ---
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    dtm_images_dir = "exp/dtm_images_csf"
    stitched_output_dir = "stitched_images"
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