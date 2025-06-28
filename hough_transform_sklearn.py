#!/usr/bin/env python3
"""
Hough Transform Image Analysis Tool (Scikit-Image Version)

This script performs Hough transforms on PNG images to detect:
1. Circles
2. Straight lines
3. Rectilinear patterns (grid-like structures)

Requirements:
    pip install scikit-image

Usage:
    python hough_transform_sklearn.py input_image.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Optional

# Scikit-image imports
from skimage import io, color, filters, feature, transform, draw, morphology
from skimage.filters import gaussian
from skimage.util import img_as_ubyte


class HoughTransformAnalyzerSklearn:
    """A class to perform various Hough transforms on images using scikit-image."""
    
    def __init__(self, image_path: str):
        """
        Initialize with an image path.
        
        Args:
            image_path: Path to the input PNG image
        """
        self.image_path = image_path
        self.original_image = None
        self.gray_image = None
        self.edges = None
        self.load_image()
    
    def load_image(self) -> None:
        """Load and preprocess the image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        # Load image (scikit-image loads as RGB by default)
        self.original_image = io.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        # Handle different image types
        if len(self.original_image.shape) == 3:
            # Convert to grayscale
            self.gray_image = color.rgb2gray(self.original_image)
        else:
            # Already grayscale
            self.gray_image = self.original_image.copy()
            # Convert grayscale to RGB for visualization
            self.original_image = color.gray2rgb(self.original_image)
        
        # Apply Gaussian blur to reduce noise
        blurred = gaussian(self.gray_image, sigma=1.5, preserve_range=True)
        
        # Edge detection using Canny
        self.edges = feature.canny(blurred, sigma=1, low_threshold=0.1, high_threshold=0.2)
    
    def detect_circles(self, min_radius: int = 10, max_radius: int = 300, 
                      num_circles: int = 10) -> Tuple[np.ndarray, List]:
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            num_circles: Maximum number of circles to detect
            
        Returns:
            Tuple of (image_with_circles, circles_list)
        """
        # Create a copy of the original image for drawing
        circles_image = self.original_image.copy()
        
        # Create radius range
        hough_radii = np.arange(min_radius, max_radius, step=2)
        
        # Apply Hough Circle Transform
        hough_res = transform.hough_circle(self.edges, hough_radii)
        
        # Select the most prominent circles
        accums, cx, cy, radii = transform.hough_circle_peaks(
            hough_res, hough_radii, total_num_peaks=num_circles,
            min_xdistance=20, min_ydistance=20
        )
        
        circles_list = []
        
        # Draw the circles
        for center_y, center_x, radius in zip(cy, cx, radii):
            # Draw the circle
            circle_perimeter = draw.circle_perimeter(center_y, center_x, radius,
                                                   shape=circles_image.shape[:2])
            circles_image[circle_perimeter] = (0, 1, 0)  # Green circle
            
            # Draw the center
            center_perimeter = draw.circle_perimeter(center_y, center_x, 2,
                                                   shape=circles_image.shape[:2])
            circles_image[center_perimeter] = (1, 0, 0)  # Red center
            
            circles_list.append((center_x, center_y, radius))
        
        return circles_image, circles_list
    
    def detect_lines(self, num_peaks: int = 100) -> Tuple[np.ndarray, List]:
        """
        Detect straight lines using Hough Line Transform.
        
        Args:
            num_peaks: Maximum number of lines to detect
            
        Returns:
            Tuple of (image_with_lines, lines_list)
        """
        # Create a copy of the original image for drawing
        lines_image = self.original_image.copy()
        
        # Standard Hough Line Transform
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = transform.hough_line(self.edges, theta=tested_angles)
        
        # Find peaks
        hough_peaks = transform.hough_line_peaks(h, theta, d, num_peaks=num_peaks,
                                               threshold=0.3*np.max(h),
                                               min_distance=20, min_angle=10)
        
        lines_list = []
        
        # Draw the lines
        for _, angle, dist in zip(*hough_peaks):
            # Convert polar coordinates to cartesian
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - lines_image.shape[1] * np.cos(angle)) / np.sin(angle)
            
            # Draw line across the image
            line_coords = draw.line(int(y0), 0, int(y1), lines_image.shape[1] - 1)
            
            # Clip coordinates to image bounds
            valid_coords = (
                (line_coords[0] >= 0) & (line_coords[0] < lines_image.shape[0]) &
                (line_coords[1] >= 0) & (line_coords[1] < lines_image.shape[1])
            )
            
            if np.any(valid_coords):
                valid_y = line_coords[0][valid_coords]
                valid_x = line_coords[1][valid_coords]
                lines_image[valid_y, valid_x] = (1, 0, 0)  # Red lines
                
                lines_list.append((dist, angle, 0, int(y0), lines_image.shape[1] - 1, int(y1)))
        
        return lines_image, lines_list
    
    def detect_line_segments(self, min_length: int = 50, max_gap: int = 10) -> Tuple[np.ndarray, List]:
        """
        Detect line segments using Probabilistic Hough Line Transform.
        
        Args:
            min_length: Minimum length of line segments
            max_gap: Maximum allowed gap between line segments
            
        Returns:
            Tuple of (image_with_line_segments, segments_list)
        """
        # Create a copy of the original image for drawing
        segments_image = self.original_image.copy()
        
        # Probabilistic Hough Line Transform
        lines = transform.probabilistic_hough_line(
            self.edges, 
            threshold=10, 
            line_length=min_length, 
            line_gap=max_gap
        )
        
        segments_list = []
        
        # Draw the line segments
        for line in lines:
            p0, p1 = line
            line_coords = draw.line(p0[1], p0[0], p1[1], p1[0])
            
            # Ensure coordinates are within image bounds
            valid_coords = (
                (line_coords[0] >= 0) & (line_coords[0] < segments_image.shape[0]) &
                (line_coords[1] >= 0) & (line_coords[1] < segments_image.shape[1])
            )
            
            if np.any(valid_coords):
                valid_y = line_coords[0][valid_coords]
                valid_x = line_coords[1][valid_coords]
                segments_image[valid_y, valid_x] = (0, 0, 1)  # Blue segments
                
                segments_list.append((p0[0], p0[1], p1[0], p1[1]))
        
        return segments_image, segments_list
    
    def detect_rectilinear_patterns(self, angle_tolerance: float = 5.0) -> Tuple[np.ndarray, dict]:
        """
        Detect rectilinear patterns by finding horizontal and vertical lines.
        
        Args:
            angle_tolerance: Tolerance for horizontal/vertical line detection (degrees)
            
        Returns:
            Tuple of (image_with_grid, pattern_info)
        """
        # Create a copy of the original image for drawing
        grid_image = self.original_image.copy()
        
        # Get line segments
        _, segments = self.detect_line_segments()
        
        horizontal_lines = []
        vertical_lines = []
        
        # Classify lines as horizontal or vertical
        for x1, y1, x2, y2 in segments:
            # Calculate angle
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angle = abs(angle)
                
                # Check if horizontal (close to 0° or 180°)
                if angle < angle_tolerance or angle > (180 - angle_tolerance):
                    horizontal_lines.append((x1, y1, x2, y2))
                    line_coords = draw.line(y1, x1, y2, x2)
                    
                    # Draw horizontal lines in green
                    valid_coords = (
                        (line_coords[0] >= 0) & (line_coords[0] < grid_image.shape[0]) &
                        (line_coords[1] >= 0) & (line_coords[1] < grid_image.shape[1])
                    )
                    if np.any(valid_coords):
                        valid_y = line_coords[0][valid_coords]
                        valid_x = line_coords[1][valid_coords]
                        grid_image[valid_y, valid_x] = (0, 1, 0)  # Green
                
                # Check if vertical (close to 90°)
                elif abs(angle - 90) < angle_tolerance:
                    vertical_lines.append((x1, y1, x2, y2))
                    line_coords = draw.line(y1, x1, y2, x2)
                    
                    # Draw vertical lines in blue
                    valid_coords = (
                        (line_coords[0] >= 0) & (line_coords[0] < grid_image.shape[0]) &
                        (line_coords[1] >= 0) & (line_coords[1] < grid_image.shape[1])
                    )
                    if np.any(valid_coords):
                        valid_y = line_coords[0][valid_coords]
                        valid_x = line_coords[1][valid_coords]
                        grid_image[valid_y, valid_x] = (0, 0, 1)  # Blue
            else:
                # Vertical line (infinite slope)
                vertical_lines.append((x1, y1, x2, y2))
                line_coords = draw.line(y1, x1, y2, x2)
                
                valid_coords = (
                    (line_coords[0] >= 0) & (line_coords[0] < grid_image.shape[0]) &
                    (line_coords[1] >= 0) & (line_coords[1] < grid_image.shape[1])
                )
                if np.any(valid_coords):
                    valid_y = line_coords[0][valid_coords]
                    valid_x = line_coords[1][valid_coords]
                    grid_image[valid_y, valid_x] = (0, 0, 1)  # Blue
        
        # Find intersections
        intersections = self._find_intersections(horizontal_lines, vertical_lines)
        
        # Draw intersections
        for x, y in intersections:
            if 0 <= int(y) < grid_image.shape[0] and 0 <= int(x) < grid_image.shape[1]:
                circle_coords = draw.disk((int(y), int(x)), 3, shape=grid_image.shape[:2])
                grid_image[circle_coords] = (1, 0, 0)  # Red intersections
        
        pattern_info = {
            'horizontal_lines': len(horizontal_lines),
            'vertical_lines': len(vertical_lines),
            'intersections': len(intersections),
            'grid_score': len(intersections) / max(1, len(horizontal_lines) + len(vertical_lines))
        }
        
        return grid_image, pattern_info
    
    def _find_intersections(self, h_lines: List, v_lines: List) -> List[Tuple[float, float]]:
        """Find intersections between horizontal and vertical lines."""
        intersections = []
        
        for hx1, hy1, hx2, hy2 in h_lines:
            for vx1, vy1, vx2, vy2 in v_lines:
                # Find intersection point
                intersection = self._line_intersection(
                    (hx1, hy1, hx2, hy2), 
                    (vx1, vy1, vx2, vy2)
                )
                if intersection:
                    x, y = intersection
                    # Check if intersection is within image bounds
                    if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
                        intersections.append((x, y))
        
        return intersections
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def analyze_all(self, save_results: bool = True) -> dict:
        """
        Perform all Hough transforms and return results.
        
        Args:
            save_results: Whether to save result images
            
        Returns:
            Dictionary containing all analysis results
        """
        print("Analyzing image with Hough transforms (scikit-image)...")
        
        # Detect circles
        circles_img, circles = self.detect_circles()
        print(f"Found {len(circles)} circles")
        
        # Detect lines
        lines_img, lines = self.detect_lines()
        print(f"Found {len(lines)} lines")
        
        # Detect line segments
        segments_img, segments = self.detect_line_segments()
        print(f"Found {len(segments)} line segments")
        
        # Detect rectilinear patterns
        grid_img, grid_info = self.detect_rectilinear_patterns()
        print(f"Grid analysis: {grid_info}")
        
        # Create visualization
        if save_results:
            self._save_results(circles_img, lines_img, segments_img, grid_img)
        
        results = {
            'circles': {
                'count': len(circles),
                'circles': circles
            },
            'lines': {
                'count': len(lines),
                'lines': lines
            },
            'segments': {
                'count': len(segments),
                'segments': segments
            },
            'grid': grid_info
        }
        
        return results
    
    def _save_results(self, circles_img: np.ndarray, lines_img: np.ndarray, 
                     segments_img: np.ndarray, grid_img: np.ndarray) -> None:
        """Save all result images."""
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Create output directory
        output_dir = f"{base_name}_hough_sklearn_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results (convert to uint8 for saving)
        io.imsave(f"{output_dir}/circles_detected.png", img_as_ubyte(circles_img))
        io.imsave(f"{output_dir}/lines_detected.png", img_as_ubyte(lines_img))
        io.imsave(f"{output_dir}/segments_detected.png", img_as_ubyte(segments_img))
        io.imsave(f"{output_dir}/grid_detected.png", img_as_ubyte(grid_img))
        io.imsave(f"{output_dir}/edges.png", img_as_ubyte(self.edges))
        
        # Create combined visualization
        self._create_combined_visualization(circles_img, lines_img, segments_img, grid_img, output_dir)
        
        print(f"Results saved to {output_dir}/")
    
    def _create_combined_visualization(self, circles_img: np.ndarray, lines_img: np.ndarray,
                                     segments_img: np.ndarray, grid_img: np.ndarray, 
                                     output_dir: str) -> None:
        """Create a combined visualization of all results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image (already in RGB)
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Edges
        axes[0, 1].imshow(self.edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection (Canny)')
        axes[0, 1].axis('off')
        
        # Circles
        axes[0, 2].imshow(circles_img)
        axes[0, 2].set_title('Circle Detection (Hough)')
        axes[0, 2].axis('off')
        
        # Lines
        axes[1, 0].imshow(lines_img)
        axes[1, 0].set_title('Line Detection (Hough)')
        axes[1, 0].axis('off')
        
        # Line segments
        axes[1, 1].imshow(segments_img)
        axes[1, 1].set_title('Line Segments (Probabilistic)')
        axes[1, 1].axis('off')
        
        # Rectilinear patterns
        axes[1, 2].imshow(grid_img)
        axes[1, 2].set_title('Rectilinear Patterns')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/combined_analysis_sklearn.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the Hough transform analysis."""
    parser = argparse.ArgumentParser(description='Perform Hough transform analysis on PNG images using scikit-image')
    parser.add_argument('image_path', help='Path to the input PNG image')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save result images')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = HoughTransformAnalyzerSklearn(args.image_path)
        
        # Perform analysis
        results = analyzer.analyze_all(save_results=not args.no_save)
        
        # Print summary
        print("\n" + "="*60)
        print("HOUGH TRANSFORM ANALYSIS SUMMARY (SCIKIT-IMAGE)")
        print("="*60)
        print(f"Circles detected: {results['circles']['count']}")
        print(f"Lines detected: {results['lines']['count']}")
        print(f"Line segments detected: {results['segments']['count']}")
        print(f"Horizontal lines: {results['grid']['horizontal_lines']}")
        print(f"Vertical lines: {results['grid']['vertical_lines']}")
        print(f"Grid intersections: {results['grid']['intersections']}")
        print(f"Grid score: {results['grid']['grid_score']:.3f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 