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

## Results
I've run an experiment over the Belterra region, satisfying the parameters of checkpoint 2.

See `experiment_TAP_2_w_insights` for the outputs of the full experiment run (focused on the Belterra region)
Total anomaly count produced by experiment: 44

For example, see:
https://github.com/samuelzxu/oaiarch/blob/ckpt_2_submission/experiment_TAP_2_w_insights/TAP_A02_2012_analysis.txt
https://github.com/samuelzxu/oaiarch/blob/ckpt_2_submission/experiment_TAP_2_w_insights/TAP_A03_2018_analysis.txt
https://github.com/samuelzxu/oaiarch/blob/ckpt_2_submission/experiment_TAP_2_w_insights/TAP_A04_2008_analysis.txt
https://github.com/samuelzxu/oaiarch/blob/ckpt_2_submission/experiment_TAP_2_w_insights/TAP_A05_2008_analysis.txt

Datasets:
- [Lidar surveys over the amazon from 2008-2018](https://daac.ornl.gov/CMS/guides/LiDAR_Forest_Inventory_Brazil.html)
- Opentopography 30m DEM (couldn't get access to higher res.)
- Sentinel L2A: NIR and Visual bands (10m res.)

Model: `o4-mini`

Footprint output example from `experiment_TAP_2_w_insights/TAP_A01_2008_analysis.txt`:
```
"anomaly_1": {
    "description": "A faint, straight ridge trending NW–SE, ~150 m long and ~0.5 m above adjacent terrain. The ridge cross‐section is trapezoidal, suggesting anthropogenic construction (levee, pathway, or causeway fill).",
    "location": {
        "lat": -2.85690,
        "lon": -54.95430,
        "radius": 75
    }
},
"anomaly_2": {
    "description": "A near‐circular arrangement of four low mounds (5–7 m diameter each) spaced around a ~25 m diameter ring. In the LiDAR DTM, these form a distinctive ‘donut’ pattern; in NIR they correlate with slightly higher reflectance (possible charcoal‐rich soils).",
    "location": {
        "lat": -2.85710,
        "lon": -54.95455,
        "radius": 15
    }
},
"anomaly_3": {
    "description": "An elongated, shallow ditch‐like depression (~2 m wide, ~80 m long) trending E–W. In the cloth‐model DTM it appears unnaturally straight with consistent width—potentially a small drainage or agricultural furrow.",
    "location": {
        "lat": -2.85675,
        "lon": -54.95460,
        "radius": 40
    }
}
```

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
