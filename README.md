# Checkpoint 2 Submission

Requirements:
```
1. Loads two independent public sources (e.g., GEDI + TerraBrasilis polygons)
2. Produces at least five candidate “anomaly” footprints (bbox WKT or lat/lon center + radius)
3. Logs all dataset IDs and OpenAI prompts. Verify: Automated script re‑runs → same five
footprints ±50 m.
4. Show us how you can use this new data in future discovery - re-prompt the model with this
leverage.
```
Total anomaly count produced: 44

I've used the following datasets:
- [Lidar surveys over the amazon from 2008-2018](https://daac.ornl.gov/CMS/guides/LiDAR_Forest_Inventory_Brazil.html)
- Opentopography 30m DEM (couldn't get access to higher res.)
- Sentinel L2A: NIR and Visual bands

The only model I've used is `o3-mini`.

My workflow is:
1. Download lidar tiles from ORNL database
2. Use Cloth Simulation Filter to produce DTMs from the point clouds (`process_lidar.py`)
3. Stitch the point clouds together so I have high-res region DTMs and time of survey (`stitch_dtms.py`)
4. For each stitched lidar file, do the following (`analyze.py`):
    - extract coordinates from dataset-provided `.kmz` file
    - download sentinel & opentopo data from the same lon, lat bounding box
    - use google's reverse geocoding API to determine the full name of the address
    - RAG: stitch together a neat prompt with some output formatting
    - prompt & pray!

Opportunities for improvement & Next Steps:
- The stitching function isn't totally accurate right now
- I could improve the quality of the DTM-from-lidar generation
- I should explore surrounding areas of high anomaly density
- I could leverage oral histories & records


To reproduce, this should work:
```
# python 3.10.14
python -m pip install -r requirements.txt
python analyze.py
```
