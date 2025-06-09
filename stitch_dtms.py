import os
import json
import re
import glob
import numpy as np
from PIL import Image
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

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

def stitch_group_images(group_key: str, polygon_names: List[str], dtm_dir: str, output_dir: str):
    """Stitches all DTM images for a given group of polygons."""
    print(f"\nStitching group: {group_key}...")
    
    group_metadata = []
    for name in polygon_names:
        file_stem = os.path.splitext(name)[0]
        metadata_path = os.path.join(dtm_dir, f"{file_stem}_dtm_csf.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                group_metadata.append(json.load(f))
        else:
            print(f"  - Warning: Metadata for {name} not found. Skipping.")

    if not group_metadata:
        print(f"No DTM images found for group {group_key}. Skipping.")
        return

    # Determine the overall bounding box for the entire group
    global_min_x = min(m['bounds']['min_x'] for m in group_metadata)
    global_max_x = max(m['bounds']['max_x'] for m in group_metadata)
    global_min_y = min(m['bounds']['min_y'] for m in group_metadata)
    global_max_y = max(m['bounds']['max_y'] for m in group_metadata)
    
    # Assuming resolution is consistent across all tiles
    resolution = group_metadata[0]['resolution']

    # Calculate the size of the stitched image
    stitched_width = int(np.ceil((global_max_x - global_min_x) / resolution))
    stitched_height = int(np.ceil((global_max_y - global_min_y) / resolution))
    
    print(f"  - Creating canvas of size: {stitched_width} x {stitched_height} pixels.")
    
    # Create a blank canvas. We use a float array for NaN support.
    stitched_array = np.full((stitched_height, stitched_width), np.nan, dtype=np.float32)

    for metadata in group_metadata:
        raw_data_path = metadata.get('dtm_raw_data')
        if not raw_data_path or not os.path.exists(raw_data_path):
            print(f"  - Warning: Raw DTM data for {metadata['lidar_file']} not found. Skipping.")
            continue
            
        dtm_array = np.load(raw_data_path)
        
        h, w = dtm_array.shape
        bounds = metadata['bounds']
        
        # Calculate pixel offsets on the main canvas
        x_offset = int(np.round((bounds['min_x'] - global_min_x) / resolution))
        y_offset = int(np.round((bounds['min_y'] - global_min_y) / resolution))
        
        # Paste the DTM into the correct location on the canvas
        # The y-axis needs to be flipped because array indices start from the top,
        # but our coordinates start from the bottom ('lower' origin).
        y_pos = stitched_height - (y_offset + h)
        
        # Ensure we don't try to write outside the canvas
        if y_pos < 0: y_pos = 0
        
        # Combine the images, only overwriting the "no data" areas
        # This basic pasting assumes non-overlapping tiles. For overlaps, more complex blending would be needed.
        target_slice = stitched_array[y_pos:y_pos+h, x_offset:x_offset+w]
        # We paste the new tile only where the canvas is currently empty (NaN)
        target_slice[np.isnan(target_slice)] = dtm_array[np.isnan(target_slice)]

    # Normalize the array to 0-255 for saving as an image
    # We ignore the NaN "no data" values during normalization
    valid_pixels = stitched_array[~np.isnan(stitched_array)]
    if valid_pixels.size > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            # Normalize to 0-255
            normalized_array = (stitched_array - min_val) * (255.0 / (max_val - min_val))
            # Set "no data" areas (which are still NaN) to black
            normalized_array[np.isnan(normalized_array)] = 0
        else:
            # Handle case where all valid pixels have the same value
            normalized_array = np.full(stitched_array.shape, 128)
            normalized_array[np.isnan(stitched_array)] = 0
    else:
        # Handle case where there are no valid pixels at all
        normalized_array = np.zeros(stitched_array.shape)

    final_image = Image.fromarray(normalized_array.astype(np.uint8), mode='L')
    
    output_path = os.path.join(output_dir, f"{group_key}_stitched.png")
    os.makedirs(output_dir, exist_ok=True)
    final_image.save(output_path)
    
    print(f"  -> Saved stitched image to {output_path}")

def main():
    """Main function to run the stitching process."""
    # --- Configuration ---
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    dtm_images_dir = "dtm_images_csf"
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

    # 1. Read polygon names from KMZ
    print("1. Parsing KMZ file to get polygon layout...")
    kml_content = extract_kmz_content(kmz_file_path)
    polygon_names = extract_polygon_names_from_kml(kml_content)
    print(f"   Found {len(polygon_names)} polygons in the KMZ file.")

    # 2. Group polygons
    print("\n2. Grouping contiguous polygons...")
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
            output_dir=stitched_output_dir
        )
        
    print("\n" + "=" * 40)
    print("Stitching process complete.")
    print(f"All stitched images are saved in '{os.path.abspath(stitched_output_dir)}'.")


if __name__ == "__main__":
    main() 