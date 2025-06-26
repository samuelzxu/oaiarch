"""
Download a GeoTIFF tile from OpenTopography and display it.

Dependencies
------------
pip install requests rasterio matplotlib pystac-client odc-stac xarray pandas numpy contextily folium pillow
# Optional (for better OSM map rendering):
pip install selenium
# If using selenium, you'll also need ChromeDriver installed
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from display_kmz import extract_polygons_from_kml, extract_kmz_content, get_polygon_bounds_from_single_polygon, get_polygon_bounds, get_item_name_from_filename
from query_sentinel_data import SentinelSTACDownloader
from datetime import datetime, timedelta
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
import contextily as ctx
import folium
from PIL import Image, ImageDraw

from pyproj import Transformer
from dotenv import load_dotenv
import os
load_dotenv()

from typing import Tuple, List, Dict, Any
import base64
from openai import OpenAI
import os
import requests
import json

def raster_to_png_data_url(
    tif_path: Path, size: Tuple[int, int] = (512, 512), save_png: bool = True
) -> str:
    """
    Down-sample the first band of the GeoTIFF, scale to 8-bit, and
    return a data-URL (PNG, base-64) suitable for the OpenAI vision API.
    Optionally saves the PNG file alongside the source TIF.

    Parameters
    ----------
    tif_path : Path
        Path to the input GeoTIFF file
    size : Tuple[int, int]
        Size of the output image in pixels
    save_png : bool
        If True, saves the PNG file next to the source TIF

    Returns
    -------
    str
        Base64 encoded PNG data URL
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1, masked=True).astype(np.float32)

    # Simple min-max stretch → 0-255
    z_min, z_max = np.nanmin(data), np.nanmax(data)
    arr = np.clip((data - z_min) / (z_max - z_min + 1e-9) * 255, 0, 255).astype(
        np.uint8
    )

    # Resize to something manageable for the API
    fig = plt.figure(frameon=False)
    fig.set_size_inches(*[s / 100 for s in size])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap="gray", origin="upper")
    
    # Save to both buffer and file if requested
    buf = io.BytesIO()
    if save_png:
        png_path = tif_path.with_suffix('.png')
        fig.savefig(png_path, format="png", dpi=100, bbox_inches='tight', pad_inches=0)
        print(f"Saved PNG to {png_path}")
    
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    png_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{png_base64}"

def png_to_data_url(png_path: Path)-> str:
    """
    Convert a PNG file to a data-URL (base-64) suitable for the OpenAI vision API.
    """
    with open(png_path, "rb") as f:
        png_base64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{png_base64}" 

def construct_rev_geocode_url(
    lat: float, lon: float, api_key: str) -> str:
    return f"https://maps.googleapis.com/maps/api/geocode/json?latlng={round(lat, 6)},{round(lon, 6)}&key={api_key}"

def extract_human_readable_from_geocode_data(geocode_data: dict) -> str:
    """
    Extract a human-readable address from the reverse geocoding data.
    """
    allowed_types = ["administrative_area_level_2","political","administrative_area_level_1"]
    if "results" in geocode_data and len(geocode_data["results"]) > 0:
        for result in geocode_data["results"]:
            components = result["address_components"]
            if "formatted_address" in result and sum(1 for c in components if set(c["types"]) & set(allowed_types)) > 2:
                print(f"Found valid formatted address: {result['formatted_address']}")
                return result["formatted_address"]
        return geocode_data["results"][0]["formatted_address"]
    return False

def analyse_with_openai(
    image_data_urls: list[str],
    model: str = "o4-mini",
    lat: float = 0.0,
    lon: float = 0.0,
    prompt_append: str = ""
) -> str:
    """
    Send images + prompt to GPT-4v's vision endpoint and return the text.
    Expects four images:
    1. OpenTopo DEM
    2. LiDAR-derived DTM
    3. Sentinel-2 Visual
    4. Sentinel-2 NIR
    """
    geocode_data = requests.get(
        construct_rev_geocode_url(lat, lon, os.environ.get("GOOGLE_EARTH_API_KEY"))
    ).json()
    loc_str = extract_human_readable_from_geocode_data(geocode_data)
    if not loc_str:
        loc_str = "State of Amazonas, Brazil"
    print(f"Reverse geocoding {lat}, {lon} → \n{loc_str}")
    prompt = f'''Analyze these images to identify and describe in detail the major terrain and geomorphological features. 
Search carefully for any archaeologically interesting anomalies.

The images are from the region of {loc_str}, at ({lat}, {lon}).

If anomalies are found:
  1. Draw on your knowledge of the region ({loc_str}) and its history.
  2. Explain how the anomaly fits into the broader historical context.
  3. Discuss how it might challenge or advance current theories in the field. Give detailed background information on the geographic, historical, and archaeological context of the region.
  4. Note any differences or correlations between the different data sources.
  5. Provide actionable insights. These insights will be used as leverage for future discovery of sites, that is, it will be appended to future prompts to AI models. To that end, provide generalizable insights that can be applied to other regions and datasets.

Target your analysis to technology and archaeology enthusiasts interested in realistic, evidence-based advancements. Provide as much detail as possible.

Always include your actionable insights inside triple brackets:

[[[
actionable insights that can be used to discover new sites, such as:
- "Note the presence of a large, flat area at the coordinates (lat, lon) that could indicate an ancient settlement."
- "The presence of a linear feature in the LiDAR data suggests a possible ancient road or pathway."
- "The NIR band shows unusual vegetation patterns that may indicate buried structures or features."
]]]

If you detect an anomaly, output:
```anomalies
"anomaly_1": {{
    "description": "A detailed description of the anomaly",
    "location": {{
        "lat": lat, 
        "lon": lon, 
        "radius": r # radius in meters
        }},  
}},
"anomaly_2": {{
    "description": "A detailed description of the anomaly",
    "location": {{
        "lat": lat, 
        "lon": lon, 
        "radius": r # radius in meters
        }},  
}},
...
"anomaly_N": {{
    "description": "A detailed description of the anomaly",
    "location": {{
        "lat": lat, 
        "lon": lon, 
        "radius": r # radius in meters
        }},  
}},
```\n\n'''+ prompt_append +"\n\n"

    client = OpenAI()  # picks up OPENAI_API_KEY from env
    print("→ Contacting OpenAI …")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in image_data_urls],
                ],
            }
        ],
    )
    return {
        'response':completion.choices[0].message.content.strip(),
        'prompt': prompt,
    }


def convert_utm_to_wgs84(utm_x, utm_y):
    # Define the transformer: from UTM zone 19S (EPSG:32719) to WGS 84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

    # Convert to latitude and longitude
    lon, lat = transformer.transform(utm_x, utm_y)

    return lon, lat


def fetch_raster_tile_from_opentopography(
    api_key: str,
    dataset: str,
    north: float,
    south: float,
    east: float,
    west: float,
    dest: Path = Path("tile.tif"),
    source: str = "globaldem"
) -> Path:
    """
    Download a GeoTIFF tile from OpenTopography's Point-Cloud API.

    Returns
    -------
    Path
        Path to the downloaded `.tif` file (possibly extracted from a ZIP).
    """
    outputFormat = "GTiff"
    ds_key = ""
    if source == "usgsdem":
        ds_key = "datasetName"
    elif source == "globaldem":
        ds_key = "demtype"
    else:
        raise ValueError(f"Invalid source: {source}")
    url = f"https://portal.opentopography.org/API/{source}?{ds_key}={dataset}&north={str(north)}&south={str(south)}&east={str(east)}&west={str(west)}&outputFormat={outputFormat}&API_Key={api_key}"

    # print("Requesting data from ", url)

    r = requests.get(url,  timeout=300)
    if (len(r.content) < 200):
        print(r.content)
    r.raise_for_status()

    # The service may send back either a bare .tif or a ZIP archive.
    content_type = r.headers.get("content-type", "")
    if "zip" in content_type or r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            tif_name = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
            zf.extract(tif_name, dest.parent)
            dest = dest.parent / tif_name
    else:
        dest.write_bytes(r.content)

    print(f"Saved raster tile to {dest.resolve()}")
    return dest


def display_raster(path: Path) -> None:
    """Render the GeoTIFF elevation raster with matplotlib."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)  # first band
        bounds = src.bounds

    fig, ax = plt.subplots(figsize=(20, 20))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(
        data, cmap="terrain", extent=extent, origin="upper"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{path.name}  –  Elevation (m)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Elevation (m)")
    plt.tight_layout()
    plt.show()

def fetch_sentinel_data(north: float, south: float, east: float, west: float, output_dir: str, prefix: str) -> Tuple[str, str]:
    """
    Fetch Sentinel-2 data for the specified bounds and return paths to visual and NIR band images.
    
    Args:
        north, south, east, west: Bounding box coordinates
        output_dir: Directory to save the Sentinel data
        prefix: Prefix for output files
        
    Returns:
        Tuple[str, str]: Paths to (visual_band_png, nir_band_png)
    """
    # Initialize downloader
    downloader = SentinelSTACDownloader()
    
    # Set date range to last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    
    # Search for products
    search_results = downloader.search_items(
        north=north,
        south=south,
        east=east,
        west=west,
        start_date=start_date,
        end_date=end_date,
        cloud_cover_max=3
    )
    
    # Check if any items were found
    items_list = list(search_results.items())
    if not items_list:
        raise ValueError("No Sentinel-2 products found for the specified area and time range")
    
    # Create geometry for data loading
    geometry = downloader.create_geometry(north, south, east, west)
    
    # Load only visual and NIR bands
    downloader.target_bands = ['visual', 'nir', 'swir']
    data = downloader.load_data(
        search_results=search_results,
        geometry=geometry,
        resolution=10
    )
    
    if data is None:
        raise ValueError("Failed to load Sentinel-2 data")
    
    # Apply scaling corrections
    scaled_data = downloader.apply_scaling(data, search_results)
    
    # Save the data
    saved_files = downloader.save_data(
        data=scaled_data,
        output_dir=output_dir,
        prefix=prefix
    )
    
    # Get the most recent visual and NIR band PNGs
    time_steps = list(data.time.values)

    # Iterate through the time steps, most recent first, until we find one we downloaded.
    if not time_steps:
        raise ValueError("No time steps found in the data")
    
    latest_time = None
    for t in reversed(time_steps):
        saved_id_nir = f"nir_{pd.to_datetime(t).strftime('%Y%m%d')}_png"
        saved_id_visual = f"visual_{pd.to_datetime(t).strftime('%Y%m%d')}_png"
        if saved_id_nir in saved_files and saved_id_visual in saved_files:
            latest_time = t
            break
    
    if latest_time is None:
        raise ValueError("No valid time step found in the saved files")
    # Format the time string for file naming
    time_str = pd.to_datetime(latest_time).strftime('%Y%m%d')
    
    visual_png = saved_files.get(f"visual_{time_str}_png")
    nir_png = saved_files.get(f"nir_{time_str}_png")
    
    if not visual_png or not nir_png:
        raise ValueError("Failed to save visual or NIR band images")
    
    return visual_png, nir_png

def fetch_osm_data(north: float, south: float, east: float, west: float, output_dir: str, prefix: str) -> str:
    """
    Fetch OpenStreetMap data for the specified bounds and return path to the rendered PNG.
    
    Args:
        north, south, east, west: Bounding box coordinates
        output_dir: Directory to save the OSM data
        prefix: Prefix for output files
        
    Returns:
        str: Path to the OSM map PNG file
    """
    import folium
    import io
    from PIL import Image
    import time
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate center point
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        tiles='OpenStreetMap',
        zoom_start=13
    )
    
    # Fit the map to the bounding box
    m.fit_bounds([[south, west], [north, east]])
    
    # Generate output filename
    osm_png_path = os.path.join(output_dir, f"{prefix}_osm.png")
    
    # Save map as HTML first
    html_path = osm_png_path.replace('.png', '.html')
    m.save(html_path)
    
    try:
        # Try to use selenium to render the map as PNG (if available)
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # Set up Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1024,768')
        
        # Create driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the HTML file
        driver.get(f'file://{os.path.abspath(html_path)}')
        
        # Wait for the map to load
        time.sleep(3)
        
        # Take screenshot
        driver.save_screenshot(osm_png_path)
        driver.quit()
        
        print(f"Successfully saved OSM map to {osm_png_path}")
        
    except ImportError:
        print("Selenium not available, using alternative method with contextily...")
        # Fallback to contextily if selenium is not available
        try:
            import contextily as ctx
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # Create a simple plot with contextily basemap
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Set the extent
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            
            # Add contextily basemap
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
            
            # Remove axes for cleaner image
            ax.set_axis_off()
            
            # Save the plot
            plt.savefig(osm_png_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Successfully saved OSM map using contextily to {osm_png_path}")
            
        except Exception as e:
            print(f"Error creating OSM map with contextily: {e}")
            # Create a simple placeholder image
            img = Image.new('RGB', (1024, 768), color='white')
            img.save(osm_png_path)
            print(f"Created placeholder OSM image at {osm_png_path}")
    
    except Exception as e:
        print(f"Error creating OSM map with selenium: {e}")
        # Fallback to contextily
        try:
            import contextily as ctx
            import matplotlib.pyplot as plt
            
            # Create a simple plot with contextily basemap
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Set the extent  
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            
            # Add contextily basemap
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
            
            # Remove axes for cleaner image
            ax.set_axis_off()
            
            # Save the plot
            plt.savefig(osm_png_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Successfully saved OSM map using contextily to {osm_png_path}")
            
        except Exception as e2:
            print(f"Error creating OSM map with contextily: {e2}")
            # Create a simple placeholder image
            img = Image.new('RGB', (1024, 768), color='white')
            img.save(osm_png_path)
            print(f"Created placeholder OSM image at {osm_png_path}")
    
    # Clean up HTML file
    if os.path.exists(html_path):
        os.remove(html_path)
    
    return osm_png_path

def parse_anomalies_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse anomalies from the AI response text.
    
    Args:
        response_text: The full response text from the AI
        
    Returns:
        List of anomaly dictionaries with description, lat, lon, and radius
    """
    anomalies = []
    
    # Look for the anomalies code block
    anomaly_pattern = r'```anomalies\s*(.*?)```'
    match = re.search(anomaly_pattern, response_text, re.DOTALL)
    
    if not match:
        return anomalies
    
    anomaly_text = match.group(1).strip()
    
    # Parse individual anomalies using regex
    # Look for "anomaly_N": { ... },
    individual_anomaly_pattern = r'"anomaly_\d+"\s*:\s*\{([^}]+)\}'
    
    for anomaly_match in re.finditer(individual_anomaly_pattern, anomaly_text):
        anomaly_content = anomaly_match.group(1)
        
        try:
            # Extract description
            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', anomaly_content)
            description = desc_match.group(1) if desc_match else "Unknown anomaly"
            
            # Extract location coordinates
            lat_match = re.search(r'"lat"\s*:\s*([-\d.]+)', anomaly_content)
            lon_match = re.search(r'"lon"\s*:\s*([-\d.]+)', anomaly_content)
            radius_match = re.search(r'"radius"\s*:\s*(\d+)', anomaly_content)
            
            if lat_match and lon_match and radius_match:
                anomaly = {
                    'description': description,
                    'lat': float(lat_match.group(1)),
                    'lon': float(lon_match.group(1)),
                    'radius': int(radius_match.group(1))
                }
                anomalies.append(anomaly)
                
        except Exception as e:
            print(f"Error parsing anomaly: {e}")
            continue
    
    return anomalies

def convert_latlon_to_pixels(lat: float, lon: float, bounds: Dict[str, float], image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert latitude/longitude coordinates to pixel coordinates within an image.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate  
        bounds: Dictionary with keys 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        image_size: Tuple of (width, height) in pixels
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    width, height = image_size
    
    # Normalize coordinates to 0-1 range
    x_norm = (lon - bounds['min_lon']) / (bounds['max_lon'] - bounds['min_lon'])
    y_norm = (bounds['max_lat'] - lat) / (bounds['max_lat'] - bounds['min_lat'])  # Flip Y axis
    
    # Convert to pixel coordinates
    x_pixel = int(x_norm * width)
    y_pixel = int(y_norm * height)
    
    # Clamp to image bounds
    x_pixel = max(0, min(x_pixel, width - 1))
    y_pixel = max(0, min(y_pixel, height - 1))
    
    return x_pixel, y_pixel

def meters_to_pixels(radius_meters: float, bounds: Dict[str, float], image_size: Tuple[int, int]) -> int:
    """
    Convert a radius in meters to pixels based on the geographic bounds and image size.
    
    Args:
        radius_meters: Radius in meters
        bounds: Geographic bounds dictionary
        image_size: Image size in pixels (width, height)
        
    Returns:
        Radius in pixels
    """
    width, height = image_size
    
    # Calculate degrees per pixel (rough approximation)
    lat_range = bounds['max_lat'] - bounds['min_lat']
    lon_range = bounds['max_lon'] - bounds['min_lon']
    
    # Use average latitude for more accurate conversion
    avg_lat = (bounds['max_lat'] + bounds['min_lat']) / 2
    
    # Approximate meters per degree at this latitude
    meters_per_degree_lat = 111320  # Roughly constant
    meters_per_degree_lon = 111320 * np.cos(np.radians(avg_lat))
    
    # Convert radius to degrees
    radius_deg_lat = radius_meters / meters_per_degree_lat
    radius_deg_lon = radius_meters / meters_per_degree_lon
    
    # Convert to pixels (use average of lat/lon pixel scales)
    radius_pixels_lat = (radius_deg_lat / lat_range) * height
    radius_pixels_lon = (radius_deg_lon / lon_range) * width
    
    # Use average and ensure minimum visible size
    radius_pixels = int((radius_pixels_lat + radius_pixels_lon) / 2)
    return max(radius_pixels, 5)  # Minimum 5 pixel radius for visibility

def draw_circles_on_image(image_path: str, anomalies: List[Dict[str, Any]], bounds: Dict[str, float]) -> None:
    """
    Draw circles on an image for each anomaly.
    
    Args:
        image_path: Path to the PNG image file
        anomalies: List of anomaly dictionaries
        bounds: Geographic bounds of the image
    """
    if not anomalies or not os.path.exists(image_path):
        return
    
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    for i, anomaly in enumerate(anomalies):
        try:
            # Convert lat/lon to pixel coordinates
            x, y = convert_latlon_to_pixels(
                anomaly['lat'], 
                anomaly['lon'], 
                bounds, 
                (width, height)
            )
            
            # Convert radius to pixels
            radius_pixels = meters_to_pixels(
                anomaly['radius'], 
                bounds, 
                (width, height)
            )
            
            # Choose color (cycle through a few colors)
            colors = ['red', 'yellow', 'cyan', 'magenta', 'orange', 'lime']
            color = colors[i % len(colors)]
            
            # Draw the circle
            draw.ellipse(
                [x - radius_pixels, y - radius_pixels, x + radius_pixels, y + radius_pixels],
                outline=color,
                width=3
            )
            
            # Draw a small center dot
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
            
            print(f"Drew circle for anomaly {i+1} at ({x}, {y}) with radius {radius_pixels}px")
            
        except Exception as e:
            print(f"Error drawing circle for anomaly {i+1}: {e}")
    
    # Save the modified image
    img.save(image_path)
    print(f"Updated {image_path} with {len(anomalies)} anomaly circles")

def add_anomaly_circles_to_images(exp_name: str, item_name: str, anomalies: List[Dict[str, Any]], bounds: Dict[str, float]) -> None:
    """
    Add anomaly circles to all available images for a processed item.
    
    Args:
        exp_name: Experiment directory name
        item_name: Name of the processed item
        anomalies: List of parsed anomalies
        bounds: Geographic bounds of the area
    """
    if not anomalies:
        return
        
    print(f"Adding {len(anomalies)} anomaly circles to images for {item_name}")
    
    # List of possible image types to annotate
    image_types = ['lidar', 'visual', 'nir', 'osm']
    
    # Try to find OpenTopography image (could have different naming)
    opentopo_candidates = [
        f"{exp_name}/{item_name}_opentopo.png",
        f"{exp_name}/{item_name}_opentopo.tif.png"  # In case it was converted
    ]
    
    for candidate in opentopo_candidates:
        if os.path.exists(candidate):
            draw_circles_on_image(candidate, anomalies, bounds)
            break
    
    # Process other image types
    for img_type in image_types:
        img_path = f"{exp_name}/{item_name}_{img_type}.png"
        draw_circles_on_image(img_path, anomalies, bounds)

def process_single_file(filename: str, exp_name: str, polygons_object, stitched_images_dir, prev_insights: str = "", anom_ct: int = 0) -> str:
    """
    Process a single file, gathering all available data sources and performing analysis.
    
    Args:
        filename: Name of the file to process
        polygons_dict: Dictionary of polygons from KMZ file
    """
    print(f"Processing {filename}...")
    
    # Skip if analysis already exists
    if os.path.exists(f"{exp_name}/{filename}_analysis.txt"):
        print(f"Analysis for {filename} already exists, skipping...")
        return

    # Get polygon bounds
    # bounds = get_polygon_bounds_from_single_polygon(polygons_dict[filename[:-8]])
    item_name = get_item_name_from_filename(filename)
    print(f"Item name: {item_name}")
    bounds = get_polygon_bounds(item_name, polygons_object)
    north = bounds["max_lat"]+0.01
    south = bounds["min_lat"]-0.01
    east = bounds["max_lon"]+0.01
    west = bounds["min_lon"]-0.01

    if north == -180 or south == 180 or east == -180 or west == 180:
        print(f"Error: Invalid bounds for {filename}. Skipping...")
        return prev_insights, anom_ct
    print(f"Bounds for {filename}: N={north}, S={south}, E={east}, W={west}")
    
    # Initialize list to store available data URLs
    data_urls = []
    data_descriptions = []
    api_key_opentopography = os.environ.get("OPENTOPOGRAPHY_API_KEY_1")
    if not api_key_opentopography:
        print("Error: OPENTOPOGRAPHY_API_KEY_1 not set in environment variables.")
        return
    
    # 1. Try to get OpenTopo data

    try:
        dataset = "SRTMGL1"
        out_path_1 = fetch_raster_tile_from_opentopography(
            api_key_opentopography, dataset, north, south, east, west, 
            source="globaldem", 
            dest=Path(f"{exp_name}/{item_name}_opentopo.tif")
        )
        data_url_1 = raster_to_png_data_url(out_path_1)
        data_urls.append(data_url_1)
        data_descriptions.append("A high-resolution elevation raster from OpenTopography")
        print("Successfully retrieved OpenTopo data")
    except Exception as e:
        print(f"Warning: Failed to fetch OpenTopo data: {e}, retrying with new API key")
        api_key_opentopography = os.environ.get("OPENTOPOGRAPHY_API_KEY_2")
        if not api_key_opentopography:
            print("Error: OPENTOPOGRAPHY_API_KEY_2 not set in environment variables.")
            return
        try:
            dataset = "SRTMGL1"
            out_path_1 = fetch_raster_tile_from_opentopography(
                api_key_opentopography, dataset, north, south, east, west, 
                source="globaldem", 
                dest=Path(f"{exp_name}/{item_name}_opentopo.tif")
            )
            data_url_1 = raster_to_png_data_url(out_path_1)
            data_urls.append(data_url_1)
            data_descriptions.append("A high-resolution elevation raster from OpenTopography")
            print("Successfully retrieved OpenTopo data")   
        except Exception as e:
            print(f"Error: Failed to fetch OpenTopo data with second API key: {e}")

    # 2. Get LiDAR data (already downloaded)
    # png_file = f"exp/dtm_images_csf/{filename}.png"
    png_file = f"{stitched_images_dir}/{filename}"
    data_url_2 = png_to_data_url(png_file)
    data_urls.append(data_url_2)
    data_descriptions.append("A digital terrain model extracted from LiDAR data using cloth simulation")
    shutil.copy(png_file, f"{exp_name}/{item_name}_lidar.png")
    # 3. Try to get Sentinel-2 Visual and NIR data
    try:
        visual_png, nir_png = fetch_sentinel_data(
            north=north,
            south=south,
            east=east,
            west=west,
            output_dir="sentinel_dls",
            prefix=filename
        )
        
        # Add Visual band
        data_url_3 = png_to_data_url(visual_png)
        data_urls.append(data_url_3)
        data_descriptions.append("A true-color (visual) Sentinel-2 satellite image")
        shutil.copy(visual_png, f"{exp_name}/{item_name}_visual.png")
        
        # Add NIR band
        data_url_4 = png_to_data_url(nir_png)
        data_urls.append(data_url_4)
        data_descriptions.append("A near-infrared (NIR) Sentinel-2 band image")
        shutil.copy(nir_png, f"{exp_name}/{item_name}_nir.png")
        print("Successfully retrieved Sentinel-2 data")
    except Exception as e:
        print(f"Warning: Failed to fetch Sentinel data: {e}")
    
    # 4. Try to get OpenStreetMap data
    try:
        osm_png = fetch_osm_data(
            north=north,
            south=south,
            east=east,
            west=west,
            output_dir="osm_data",
            prefix=filename
        )
        
        # Add OSM data
        data_url_osm = png_to_data_url(osm_png)
        data_urls.append(data_url_osm)
        data_descriptions.append("An OpenStreetMap rendering showing roads, buildings, and other infrastructure")
        shutil.copy(osm_png, f"{exp_name}/{item_name}_osm.png")
        print("Successfully retrieved OpenStreetMap data")
    except Exception as e:
        print(f"Warning: Failed to fetch OpenStreetMap data: {e}")
    
    # Only proceed if we have at least the LiDAR data
    if len(data_urls) < 2:
        print(f"Error: Not enough data sources available for {filename}")
        return
        
    # Update the analysis prompt based on available data
    def get_prompt_suffix(descriptions: list[str]) -> str:
        # Insert the numbered list of available data sources
        numbered_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))
        return f"The images show:\n{numbered_list}\n"

    p_suffix = get_prompt_suffix(data_descriptions)
    if len(prev_insights) > 0:
        p_suffix += f"\n\nPrevious insights:\n{prev_insights}\n"
    try:
        analysis_dict = analyse_with_openai(
            data_urls,
            lat=(north+south)/2,
            lon=(east+west)/2,
            prompt_append=p_suffix,
        )
        
        print(f"Writing analysis for {filename}...")
        with open(f"{exp_name}/{item_name}_prompt.txt", "w") as f:
            f.write(f"Analysis for {item_name}\n\n")
            f.write(f"{str(analysis_dict['prompt'])}\n\n")
        with open(f"{exp_name}/{item_name}_analysis.txt", "w") as f:
            f.write(f"{str(analysis_dict['response'])}\n")
        
        # Parse anomalies from the response
        anomalies = parse_anomalies_from_response(analysis_dict['response'])
        
        # If anomalies were found, draw circles on the images
        if anomalies:
            print(f"Found {len(anomalies)} anomalies in the analysis for {filename}")
            
            # Create bounds dictionary in the format expected by the drawing functions
            image_bounds = {
                'min_lat': south,
                'max_lat': north,
                'min_lon': west,
                'max_lon': east
            }
            
            # Add circles to all available images
            add_anomaly_circles_to_images(exp_name, item_name, anomalies, image_bounds)
            
            # Save anomaly data as JSON for future reference
            anomaly_file = f"{exp_name}/{item_name}_anomalies.json"
            with open(anomaly_file, "w") as f:
                json.dump(anomalies, f, indent=2)
            print(f"Saved anomaly data to {anomaly_file}")
        else:
            print(f"No anomalies detected in the analysis for {filename}")
        
        insight_pattern = r"\[\[\[(.*?)\]\]\]"
        insights = re.findall(insight_pattern, analysis_dict['response'], re.DOTALL)
        anomalies_pattern = r'"anomaly_\d+"'
        anomalies_count = re.findall(anomalies_pattern, analysis_dict['response'])
        if anomalies_count:
            print(f"Found anomalies in the analysis for {filename}: {len(anomalies_count)} anomalies detected.")
        anom_ct += len(anomalies_count)
        if insights:
            return insights[0].strip(), anom_ct
        else:
            print(f"No actionable insights found in the analysis for {filename}.")
            return "", anom_ct
    except Exception as e:
        print(f"Error during analysis of {filename}: {e}")
        return prev_insights, anom_ct

def main():
    """Main function to process all files."""
    # Load KMZ data
    print("Extracting KML content from KMZ file...")
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    kml_content = extract_kmz_content(kmz_file_path)

    exp_name = "experiment_w_OSM"
    stitched_images_dir = "stitched_images_v6"

    if not os.path.exists(stitched_images_dir):
        print(f"Directory {stitched_images_dir} does not exist. Please check the path.")
        return
    os.makedirs(exp_name, exist_ok=True)

    print("Parsing polygons from KML...")
    polygons = extract_polygons_from_kml(kml_content)
    # polygons_dict = {
    #     polygon["name"][:-4]: polygon for polygon in polygons
    # }

    # Get list of files to process
    # all_files = list(set(list(map(
    #     lambda x: x.split('.')[0], 
    #     filter(lambda x: x.endswith('.png'), os.listdir('exp/dtm_images_csf'))
    # ))))
    all_files = list(filter(lambda x: x.endswith('.png'), os.listdir(stitched_images_dir)))

    # Process each file
    insights = ""
    anom_ct = 0
    for i, filename in enumerate(all_files):
        print(f"Processing file {i+1}/{len(all_files)}: {filename}")
        # Check if the file is already processed
        if os.path.exists(f"{exp_name}/{get_item_name_from_filename(filename)}_analysis.txt"):
            print(f"Analysis for {filename} already exists, skipping...")
            continue
        insights, anom_ct = process_single_file(filename, exp_name, polygons, anom_ct=anom_ct, prev_insights=insights, stitched_images_dir=stitched_images_dir)
        print(f"Processed {filename}, current anomaly count: {anom_ct}")


if __name__ == "__main__":
    main()

