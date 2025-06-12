"""
Download a GeoTIFF tile from OpenTopography and display it.

Dependencies
------------
pip install requests rasterio matplotlib
"""
from __future__ import annotations

import io
import sys
import shutil
import zipfile
from pathlib import Path
from display_kmz import extract_polygons_from_kml, extract_kmz_content, get_polygon_bounds_from_single_polygon, get_polygon_bounds

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
) -> str:
    """
    Send an image + prompt to GPT-4o’s vision endpoint and return the text.
    """
    geocode_data = requests.get(
        construct_rev_geocode_url(lat, lon, os.environ.get("GOOGLE_EARTH_API_KEY"))
    ).json()
    loc_str = extract_human_readable_from_geocode_data(geocode_data)
    if not loc_str:
        loc_str = "State of Amazonas, Brazil"
    print(f"Reverse geocoding {lat}, {lon} → \n{loc_str}")
    prompt = f"""Analyze these images to identify and describe in detail the major terrain and geomorphological features. 
Search carefully for any archaeologically interesting anomalies. The first image is a high-resolution elevation raster, and the second is a digital terrain model extracted from LiDAR data using cloth simulation.
The images are from the region of {loc_str}, at ({lat}, {lon}).

If anomalies are found:
  1. Draw on your knowledge of the region ({loc_str}) and its history.
  2. Explain how the anomaly fits into the broader historical context.
  3. Discuss how it might challenge or advance current theories in the field.

Target your analysis to technology and archaeology enthusiasts interested in realistic, evidence-based advancements. Provide as much detail as possible.

Always include your key takeaways inside triple brackets:

[[[
insight
]]]

If you detect an anomaly, output:
\\{r'boxed{"FOUND"}'}"""

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
        # max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


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
    Download a GeoTIFF tile from OpenTopography’s Point-Cloud API.

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

api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
all_files = list(set(list(map(lambda x: x.split('.')[0], filter(lambda x: x.endswith('.png'), os.listdir('exp/dtm_images_csf'))))))

kmz_file_path = "cms_brazil_lidar_tile_inventory.kmz"

print("Extracting KML content from KMZ file...")
kml_content = extract_kmz_content(kmz_file_path)

print("Parsing polygons from KML...")
polygons = extract_polygons_from_kml(kml_content)

polygons_dict = {
    polygon["name"][:-4]: polygon for polygon in polygons
}

for filename in all_files:
    # Check if analysis already exists - if it does, skip
    if os.path.exists(f"analysis_fine/{filename}_analysis.txt"):
        print(f"Analysis for {filename} already exists, skipping...")
        continue
    print(f"Processing {filename}...")
    png_file = f"exp/dtm_images_csf/{filename}.png"
    shutil.copy(png_file, f"out/{filename}_lidar.png")

    
    bounds = get_polygon_bounds_from_single_polygon(polygons_dict[filename[:-8]])
    north = bounds["max_lat"] + 0.05
    south = bounds["min_lat"] - 0.05
    east = bounds["max_lon"] + 0.05
    west = bounds["min_lon"] - 0.05
    dataset = "SRTMGL1"

    out_path_1 = fetch_raster_tile_from_opentopography(api_key, dataset, north, south, east, west, source="globaldem", dest=Path(f"out/{filename}_opentopo.tif"))

    data_url_1 = raster_to_png_data_url(out_path_1)
    shutil.copy('out.png', f"out/{filename}_dem.png")

    data_url_2 = png_to_data_url(png_file)

    # out_path_2 = ... query sentinel data

    analysis = analyse_with_openai([data_url_1, data_url_2], lat=(north+south)/2, lon=(east+west)/2)
    print(f"Writing analysis for {filename}...")
    with open(f"analysis_fine/{filename}_analysis.txt", "w") as f:
        f.write(analysis)

