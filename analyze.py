"""
Download a GeoTIFF tile from OpenTopography and display it.

Dependencies
------------
pip install requests rasterio matplotlib pystac-client odc-stac xarray pandas numpy
"""
from __future__ import annotations

import io
import sys
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

from pyproj import Transformer
from dotenv import load_dotenv
import os
load_dotenv()

from typing import Tuple
import base64
from openai import OpenAI
import os
import requests

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
    return f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"

def extract_human_readable_from_geocode_data(geocode_data: dict) -> str:
    """
    Extract a human-readable address from the reverse geocoding data.
    """
    allowed_types = ["administrative_area_level_2","political","administrative_area_level_1"]
    if "results" in geocode_data and len(geocode_data["results"]) > 0:
        for result in geocode_data["results"]:
            components = result["address_components"]
            if "formatted_address" in result and sum(1 for c in components if c.get("types") in allowed_types) >= 2:
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
    prompt = f"""Analyze these four images to identify and describe in detail the major terrain and geomorphological features. 
Search carefully for any archaeologically interesting anomalies. The images show:
1. A high-resolution elevation raster from OpenTopography
2. A digital terrain model extracted from LiDAR data using cloth simulation
3. A true-color (visual) Sentinel-2 satellite image
4. A near-infrared (NIR) Sentinel-2 band image

The images are from the region of {loc_str}, at ({lat}, {lon}).

If anomalies are found:
  1. Draw on your knowledge of the region ({loc_str}) and its history.
  2. Explain how the anomaly fits into the broader historical context.
  3. Discuss how it might challenge or advance current theories in the field.
  4. Note any differences or correlations between the different data sources.

Target your analysis to technology and archaeology enthusiasts interested in realistic, evidence-based advancements. Provide as much detail as possible.

Always include your key takeaways inside triple brackets:

[[[
insight
]]]

If you detect an anomaly, output:
\\{r'boxed{"FOUND"}'}"""+ prompt_append

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
        cloud_cover_max=10
    )
    
    # Check if any items were found
    items_list = list(search_results.items())
    if not items_list:
        raise ValueError("No Sentinel-2 products found for the specified area and time range")
    
    # Create geometry for data loading
    geometry = downloader.create_geometry(north, south, east, west)
    
    # Load only visual and NIR bands
    downloader.target_bands = ['visual', 'nir']
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
    latest_time = max(time_steps)
    time_str = pd.to_datetime(latest_time).strftime('%Y%m%d')
    
    visual_png = saved_files.get(f"visual_{time_str}_png")
    nir_png = saved_files.get(f"nir_{time_str}_png")
    
    if not visual_png or not nir_png:
        raise ValueError("Failed to save visual or NIR band images")
    
    return visual_png, nir_png

def process_single_file(filename: str, exp_name: str, polygons_object) -> None:
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
    north = bounds["max_lat"]
    south = bounds["min_lat"]
    east = bounds["max_lon"]
    west = bounds["min_lon"]

    if north == -180 or south == 180 or east == -180 or west == 180:
        print(f"Error: Invalid bounds for {filename}. Skipping...")
        return
    print(f"Bounds for {filename}: N={north}, S={south}, E={east}, W={west}")
    
    # Initialize list to store available data URLs
    data_urls = []
    data_descriptions = []
    
    # 1. Try to get OpenTopo data
    try:
        dataset = "SRTMGL1"
        out_path_1 = fetch_raster_tile_from_opentopography(
            api_key, dataset, north, south, east, west, 
            source="globaldem", 
            dest=Path(f"{exp_name}/{item_name}_opentopo.tif")
        )
        data_url_1 = raster_to_png_data_url(out_path_1)
        data_urls.append(data_url_1)
        data_descriptions.append("A high-resolution elevation raster from OpenTopography")
        print("Successfully retrieved OpenTopo data")
    except Exception as e:
        print(f"Warning: Failed to fetch OpenTopo data: {e}")

    # 2. Get LiDAR data (already downloaded)
    # png_file = f"exp/dtm_images_csf/{filename}.png"
    png_file = f"stitched_images/{filename}"
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
    
    # Only proceed if we have at least the LiDAR data
    if len(data_urls) < 2:
        print(f"Error: Not enough data sources available for {filename}")
        return
        
    # Update the analysis prompt based on available data
    def get_prompt_suffix(descriptions: list[str]) -> str:
        # Insert the numbered list of available data sources
        numbered_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))
        return f"The images show:\n{numbered_list}\n"

    try:
        analysis_dict = analyse_with_openai(
            data_urls,
            lat=(north+south)/2,
            lon=(east+west)/2,
            prompt_append=get_prompt_suffix(data_descriptions),
        )
        
        print(f"Writing analysis for {filename}...")
        with open(f"{exp_name}/{item_name}_prompt.txt", "w") as f:
            f.write(f"{str(analysis_dict['prompt'])}\n\n")
        with open(f"{exp_name}/{item_name}_analysis.txt", "w") as f:
            f.write(f"{str(analysis_dict['response'])}\n")
    except Exception as e:
        print(f"Error during analysis of {filename}: {e}")

def main():
    """Main function to process all files."""
    # Load KMZ data
    print("Extracting KML content from KMZ file...")
    kml_content = extract_kmz_content(kmz_file_path)

    exp_name = "exp_v4"
    os.makedirs(exp_name, exist_ok=True)

    print("Parsing polygons from KML...")
    polygons = extract_polygons_from_kml(kml_content)
    polygons_dict = {
        polygon["name"][:-4]: polygon for polygon in polygons
    }

    # Get list of files to process
    # all_files = list(set(list(map(
    #     lambda x: x.split('.')[0], 
    #     filter(lambda x: x.endswith('.png'), os.listdir('exp/dtm_images_csf'))
    # ))))
    all_files = list(filter(lambda x: x.endswith('.png'), os.listdir('stitched_images')))

    # Process each file
    for filename in all_files:
        process_single_file(filename, exp_name, polygons)

if __name__ == "__main__":
    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"
    main()

