import laspy
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import CSF
import json

def create_dtm_from_lidar(lidar_file_path: str, output_dir: str, resolution: float = 0.5):
    """
    Processes a single LiDAR file to create a Digital Terrain Model (DTM)
    using Cloth Simulation Filtering (CSF) to identify ground points.

    - Reads the LiDAR point cloud.
    - Applies CSF to classify ground and non-ground points.
    - Interpolates the ground points to create a DTM raster.
    - Saves the DTM as a high-contrast grayscale PNG image.

    Args:
        lidar_file_path: Path to the input .laz or .las file.
        output_dir: Directory to save the output DTM image.
        resolution: The resolution of the output DTM image in meters per pixel.
    """
    try:
        print(f"Processing {lidar_file_path}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Read LiDAR data
        with laspy.open(lidar_file_path) as f:
            las = f.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()

        if points.shape[0] == 0:
            print(f"Warning: No points found in {lidar_file_path}. Skipping.")
            return

        # 2. Apply Cloth Simulation Filter
        csf = CSF.CSF()
        
        # Set CSF parameters. These are the default values, but they can be tuned.
        # For forested areas, a larger cloth_resolution might be needed if the DTM is too noisy.
        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = 0.5 # The size of the grid for the cloth simulation
        csf.params.rigidness = 3 # The rigidity of the cloth
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.5 # The distance threshold for classifying points
        csf.params.interations = 500

        # Perform the filtering
        csf.setPointCloud(points)
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
        ground_indices = np.array(ground)
        
        if ground_indices.size == 0:
            print(f"Warning: CSF could not extract any ground points from {lidar_file_path}. Skipping.")
            return

        ground_points = points[ground_indices]
        print(f"  -> Found {len(ground_points)} ground points using CSF.")

        # 3. Create DTM by interpolating ground points
        min_x, min_y = np.min(ground_points[:, 0]), np.min(ground_points[:, 1])
        max_x, max_y = np.max(ground_points[:, 0]), np.max(ground_points[:, 1])

        grid_x, grid_y = np.mgrid[min_x:max_x:resolution, min_y:max_y:resolution]
        
        dtm = griddata(
            ground_points[:, :2],  # XY coordinates
            ground_points[:, 2],   # Z values
            (grid_x, grid_y),
            method='cubic' # 'cubic' for smoother terrain, 'linear' for faster results
        )
        
        # Handle areas with no data by filling with the mean of the valid data
        dtm_filled = np.nan_to_num(dtm, nan=np.nanmean(dtm))

        # 4. Save raw DTM data as .npy file
        file_name = Path(lidar_file_path).stem
        npy_output_path = os.path.join(output_dir, f"{file_name}_dtm_csf.npy")
        np.save(npy_output_path, dtm) # Save the array with NaNs
        print(f"  -> Saved raw DTM data to {npy_output_path}")

        # 5. Save a visualized DTM as a PNG image for preview
        plt.figure(figsize=(15, 15))
        # Using a high-contrast grayscale colormap which is often better for spotting subtle anomalies
        plt.imshow(dtm_filled.T, cmap='gray', origin='lower')
        plt.axis('off')
        
        png_output_path = os.path.join(output_dir, f"{file_name}_dtm_csf.png")
        plt.savefig(png_output_path, bbox_inches='tight', pad_inches=0, dpi=300) # Higher DPI for better detail
        plt.close()
        
        print(f"  -> Saved preview DTM to {png_output_path}")

        # 6. Save metadata
        metadata = {
            'lidar_file': lidar_file_path,
            'dtm_preview_image': png_output_path,
            'dtm_raw_data': npy_output_path,
            'bounds': {
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y,
            },
            'resolution': resolution,
            'ground_points': len(ground_points)
        }
        
        metadata_path = os.path.join(output_dir, f"{file_name}_dtm_csf.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"  -> Saved metadata to {metadata_path}")

    except Exception as e:
        print(f"Error processing {lidar_file_path}: {e}")

def main():
    """
    Main function to find and process all LiDAR files in a directory.
    """
    # --- Configuration ---
    # IMPORTANT: Update this path to the directory containing your .laz/.las files
    lidar_data_directory = "/mnt/datasets/lidar_nasa/orders/926ea6cc8cdee77c3ed86112e89c08f5/LiDAR_Forest_Inventory_Brazil/data"
    
    # Directory where the output DTM images will be saved
    output_directory = "dtm_images_csf"
    
    # Processing parameters
    dtm_resolution = 0.5     # DTM resolution in meters/pixel (lower is higher res)
    # ---------------------

    print("Starting LiDAR DTM Generation with Cloth Simulation Filter")
    print("=" * 60)
    print(f"Input Lidar Directory: '{os.path.abspath(lidar_data_directory)}'")
    print(f"Output Image Directory: '{os.path.abspath(output_directory)}'")
    
    if not os.path.isdir(lidar_data_directory):
        print(f"\nError: The specified lidar data directory does not exist: '{lidar_data_directory}'")
        print("Please create this directory, place your .laz files inside it, or update the 'lidar_data_directory' variable in this script.")
        return

    # Find all .laz and .las files in the specified directory
    search_paths = [
        os.path.join(lidar_data_directory, '**', '*.laz'),
        os.path.join(lidar_data_directory, '**', '*.las')
    ]
    
    lidar_files = []
    for path in search_paths:
        lidar_files.extend(glob.glob(path, recursive=True))

    if not lidar_files:
        print("\nNo .laz or .las files found. Please check the 'lidar_data_directory'.")
        return
        
    print(f"\nFound {len(lidar_files)} LiDAR files to process.\n")

    for lidar_file in lidar_files:
        create_dtm_from_lidar(
            lidar_file_path=lidar_file,
            output_dir=output_directory,
            resolution=dtm_resolution
        )

    print("\n" + "=" * 60)
    print("Processing complete.")
    print(f"All generated DTM images are saved in '{os.path.abspath(output_directory)}'.")

if __name__ == "__main__":
    main() 