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
    """Stitches all DTM images for a given group of polygons by fitting DTM data to polygon coordinates."""
    print(f"\nStitching group: {group_key}...")
    
    # Get bounds from KMZ polygon data for the entire group
    group_bounds = get_polygon_bounds(group_key, polygons)
    if group_bounds["max_lat"] == -180 or group_bounds["min_lat"] == 180 or group_bounds["max_lon"] == -180 or group_bounds["min_lon"] == 180:
        print(f"  - Warning: Invalid bounds for group {group_key}. Skipping.")
        return

    # Use a more reasonable resolution - about 5m per pixel
    # This prevents creating massive canvases
    resolution_deg = 0.00005  # approximately 5m at equator
    
    # Calculate canvas size based on polygon bounds
    width_deg = group_bounds["max_lon"] - group_bounds["min_lon"]
    height_deg = group_bounds["max_lat"] - group_bounds["min_lat"]
    
    canvas_width = int(np.ceil(width_deg / resolution_deg))
    canvas_height = int(np.ceil(height_deg / resolution_deg))
    
    print(f"  - Group bounds: lat({group_bounds['min_lat']:.6f}, {group_bounds['max_lat']:.6f}), lon({group_bounds['min_lon']:.6f}, {group_bounds['max_lon']:.6f})")
    print(f"  - Canvas size: {canvas_width} x {canvas_height} pixels")
    print(f"  - Resolution: {resolution_deg:.6f} degrees/pixel (~{resolution_deg * 111000:.1f}m at equator)")
    
    # Check for reasonable dimensions
    if canvas_width > 10000 or canvas_height > 10000:
        print(f"  - Warning: Canvas too large ({canvas_width} x {canvas_height}). Skipping group {group_key}.")
        return
    
    # Create blank canvas
    stitched_array = np.full((canvas_height, canvas_width), np.nan, dtype=np.float32)
    
    # Collect valid tiles first
    valid_tiles = []
    for name in polygon_names:
        # Find corresponding polygon
        polygon = next((p for p in polygons if p['name'] == name), None)
        if not polygon:
            print(f"  - Warning: No polygon found for {name}")
            continue
            
        # Find corresponding DTM file
        file_stem = os.path.splitext(name)[0]
        dtm_path = os.path.join(dtm_dir, f"{file_stem}_dtm_csf.npy")
        if not os.path.exists(dtm_path):
            print(f"  - Warning: DTM file not found for {name}")
            continue
            
        # Load DTM data
        try:
            dtm_array = np.load(dtm_path)
            if dtm_array.size == 0:
                print(f"  - Warning: Empty DTM array for {name}")
                continue
        except Exception as e:
            print(f"  - Warning: Failed to load DTM for {name}: {e}")
            continue
            
        # Get polygon bounds
        poly_bounds = get_polygon_bounds_from_single_polygon(polygon)
        
        valid_tiles.append({
            'name': name,
            'polygon': polygon,
            'dtm_array': dtm_array,
            'bounds': poly_bounds
        })
    
    if not valid_tiles:
        print(f"  - No valid tiles found for group {group_key}")
        return
    
    print(f"  - Processing {len(valid_tiles)} valid tiles")
    
    # Process each valid tile
    tiles_processed = 0
    for tile in valid_tiles:
        name = tile['name']
        dtm_array = tile['dtm_array']
        poly_bounds = tile['bounds']
        
        # Calculate pixel coordinates for this tile
        # Convert lat/lon bounds to pixel coordinates
        left_px = int((poly_bounds["min_lon"] - group_bounds["min_lon"]) / resolution_deg)
        right_px = int((poly_bounds["max_lon"] - group_bounds["min_lon"]) / resolution_deg)
        top_px = int((group_bounds["max_lat"] - poly_bounds["max_lat"]) / resolution_deg)
        bottom_px = int((group_bounds["max_lat"] - poly_bounds["min_lat"]) / resolution_deg)
        
        # Calculate target dimensions
        target_width = right_px - left_px
        target_height = bottom_px - top_px
        
        if target_width <= 0 or target_height <= 0:
            print(f"  - Warning: Invalid target dimensions for {name}: {target_width}x{target_height}")
            continue
        
        # Ensure we don't go outside canvas bounds
        left_px = max(0, left_px)
        top_px = max(0, top_px)
        right_px = min(canvas_width, right_px)
        bottom_px = min(canvas_height, bottom_px)
        
        actual_width = right_px - left_px
        actual_height = bottom_px - top_px
        
        if actual_width <= 0 or actual_height <= 0:
            print(f"  - Warning: Tile {name} falls outside canvas bounds")
            continue
        
        # Resize DTM to fit the target area
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_y = actual_height / dtm_array.shape[0]
            zoom_x = actual_width / dtm_array.shape[1]
            
            # Only resize if necessary (avoid unnecessary interpolation)
            if abs(zoom_y - 1.0) > 0.01 or abs(zoom_x - 1.0) > 0.01:
                resized_dtm = zoom(dtm_array, (zoom_y, zoom_x), order=1, prefilter=False)
            else:
                resized_dtm = dtm_array.copy()
            
            # Ensure exact size match
            if resized_dtm.shape != (actual_height, actual_width):
                # Crop or pad to exact size
                h, w = resized_dtm.shape
                if h > actual_height or w > actual_width:
                    resized_dtm = resized_dtm[:actual_height, :actual_width]
                elif h < actual_height or w < actual_width:
                    padded = np.full((actual_height, actual_width), np.nan, dtype=np.float32)
                    padded[:h, :w] = resized_dtm
                    resized_dtm = padded
            
        except Exception as e:
            print(f"  - Warning: Failed to resize DTM for {name}: {e}")
            continue
        
        # Place the tile on the canvas
        target_slice = stitched_array[top_px:bottom_px, left_px:right_px]
        
        # Only place pixels where the canvas is currently empty (NaN) and DTM has valid data
        mask = np.isnan(target_slice) & np.isfinite(resized_dtm)
        target_slice[mask] = resized_dtm[mask]
        
        tiles_processed += 1
        print(f"  - Placed {name} at ({left_px}, {top_px}) with size ({actual_width}, {actual_height})")
    
    if tiles_processed == 0:
        print(f"  - No tiles processed for group {group_key}")
        return
        
    # Normalize for image output
    valid_pixels = stitched_array[np.isfinite(stitched_array)]
    if valid_pixels.size > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            normalized_array = (stitched_array - min_val) * (255.0 / (max_val - min_val))
            normalized_array[~np.isfinite(normalized_array)] = 0
        else:
            normalized_array = np.full(stitched_array.shape, 128)
            normalized_array[~np.isfinite(normalized_array)] = 0
    else:
        print(f"  - Warning: No valid pixels found for group {group_key}")
        normalized_array = np.zeros(stitched_array.shape)
    
    # Save image
    final_image = Image.fromarray(normalized_array.astype(np.uint8), mode='L')
    output_path = os.path.join(output_dir, f"{group_key}_stitched.png")
    os.makedirs(output_dir, exist_ok=True)
    final_image.save(output_path)
    
    # Save location information
    info = {
        "group_key": group_key,
        "bounds_latlon": {
            "min_lat": group_bounds["min_lat"],
            "max_lat": group_bounds["max_lat"],
            "min_lon": group_bounds["min_lon"],
            "max_lon": group_bounds["max_lon"]
        },
        "center_latlon": {
            "lat": (group_bounds["min_lat"] + group_bounds["max_lat"]) / 2,
            "lon": (group_bounds["min_lon"] + group_bounds["max_lon"]) / 2
        },
        "resolution_deg": resolution_deg,
        "image_size": {
            "width": canvas_width,
            "height": canvas_height
        },
        "tiles_processed": tiles_processed
    }
    
    info_path = os.path.join(output_dir, f"{group_key}_stitched_location.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"  -> Saved stitched image to {output_path}")
    print(f"  -> Saved location info to {info_path}")
    print(f"  -> Processed {tiles_processed} tiles")
    print(f"  -> Valid pixels: {valid_pixels.size} / {stitched_array.size} ({100*valid_pixels.size/stitched_array.size:.1f}%)")

def main():
    """Main function to run the stitching process."""
    # --- Configuration ---
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    dtm_images_dir = "exp/dtm_images_csf"
    stitched_output_dir = "stitched_images_v4"
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