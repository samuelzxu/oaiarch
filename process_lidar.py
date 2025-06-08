import laspy
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def create_dtm_from_lidar(lidar_file_path: str, output_dir: str, block_size: int = 10, ground_percentile: float = 1.0, resolution: float = 0.5):
    """
    Processes a single LiDAR file to create a Digital Terrain Model (DTM).

    - Reads the LiDAR point cloud.
    - Divides the area into blocks.
    - Selects the lowest points in each block to represent the ground.
    - Interpolates these ground points to create a DTM raster.
    - Saves the DTM as a PNG image.

    Args:
        lidar_file_path: Path to the input .laz or .las file.
        output_dir: Directory to save the output DTM image.
        block_size: The size (in meters) of the grid cells for ground point filtering.
        ground_percentile: The percentage of lowest points to keep in each block (e.g., 1.0 for 1%).
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

        # 2. Grid the data and filter for lowest points
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

        ground_points = []
        
        for x0 in np.arange(min_x, max_x, block_size):
            for y0 in np.arange(min_y, max_y, block_size):
                x1, y1 = x0 + block_size, y0 + block_size
                
                # Find points within the current block
                mask = (points[:, 0] >= x0) & (points[:, 0] < x1) & \
                       (points[:, 1] >= y0) & (points[:, 1] < y1)
                
                block_points = points[mask]

                if block_points.shape[0] > 0:
                    # Calculate the number of points to keep based on the percentile
                    k = int(np.ceil(block_points.shape[0] * (ground_percentile / 100.0)))
                    k = max(1, k) # Ensure at least one point is selected
                    
                    # Find the indices of the points with the lowest Z values
                    lowest_indices = np.argpartition(block_points[:, 2], k)[:k]
                    
                    ground_points.append(block_points[lowest_indices])

        if not ground_points:
            print(f"Warning: Could not extract any ground points from {lidar_file_path}. Skipping.")
            return
            
        ground_points = np.vstack(ground_points)

        # 3. Create DTM by interpolating ground points
        grid_x, grid_y = np.mgrid[min_x:max_x:resolution, min_y:max_y:resolution]
        
        dtm = griddata(
            ground_points[:, :2],  # XY coordinates
            ground_points[:, 2],   # Z values
            (grid_x, grid_y),
            method='cubic' # Use 'cubic' for smoother terrain, 'linear' for faster
        )
        
        # Handle areas with no data
        dtm = np.nan_to_num(dtm, nan=np.nanmean(dtm))

        # 4. Save DTM as an image
        plt.figure(figsize=(15, 15))
        plt.imshow(dtm.T, cmap='terrain', origin='lower')
        plt.axis('off')
        
        file_name = Path(lidar_file_path).stem
        output_path = os.path.join(output_dir, f"{file_name}_dtm.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"  -> Saved DTM to {output_path}")

    except Exception as e:
        print(f"Error processing {lidar_file_path}: {e}")

def main():
    """
    Main function to find and process all LiDAR files in a directory.
    """
    # --- Configuration ---
    # IMPORTANT: Update this path to the directory containing your .laz/.las files
    lidar_data_directory = "lidar_data"
    
    # Directory where the output DTM images will be saved
    output_directory = "dtm_images"
    
    # Processing parameters
    block_size = 20          # Grid size in meters for filtering (e.g., 20x20m)
    ground_percentile = 1.0  # Keep the bottom 1% of points as ground
    dtm_resolution = 0.5     # DTM resolution in meters/pixel (lower is higher res)
    # ---------------------

    print("Starting LiDAR DTM Generation")
    print("=" * 40)
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
            block_size=block_size,
            ground_percentile=ground_percentile,
            resolution=dtm_resolution
        )

    print("\n" + "=" * 40)
    print("Processing complete.")
    print(f"All generated DTM images are saved in '{os.path.abspath(output_directory)}'.")

if __name__ == "__main__":
    main() 