#!/usr/bin/env python3
"""
Hough Transform Image Analysis Tool

This script performs Hough transforms on PNG images to detect:
1. Circles
2. Straight lines
3. Rectilinear patterns (grid-like structures)

Requirements:
    pip install opencv-python

Usage:
    python hough_transform.py input_image.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Optional


class HoughTransformAnalyzer:
    """A class to perform various Hough transforms on images."""
    
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
        
        # Load image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        # Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (9, 9), 2)
        
        # Edge detection using Canny
        self.edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    def detect_circles(self, min_radius: int = 10, max_radius: int = 300) -> Tuple[np.ndarray, List]:
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            Tuple of (image_with_circles, circles_list)
        """
        # Create a copy of the original image for drawing
        circles_image = self.original_image.copy()
        
        # Apply HoughCircles
        circles = cv2.HoughCircles(
            self.gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        circles_list = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Draw the circles
            for (x, y, r) in circles:
                # Draw the circle
                cv2.circle(circles_image, (x, y), r, (0, 255, 0), 2)
                # Draw the center
                cv2.circle(circles_image, (x, y), 2, (0, 0, 255), 3)
                circles_list.append((x, y, r))
        
        return circles_image, circles_list
    
    def detect_lines(self, rho: float = 1, theta: float = np.pi/180, 
                    threshold: int = 100) -> Tuple[np.ndarray, List]:
        """
        Detect straight lines using Hough Line Transform.
        
        Args:
            rho: Distance resolution in pixels
            theta: Angle resolution in radians
            threshold: Minimum number of votes
            
        Returns:
            Tuple of (image_with_lines, lines_list)
        """
        # Create a copy of the original image for drawing
        lines_image = self.original_image.copy()
        
        # Standard Hough Line Transform
        lines = cv2.HoughLines(self.edges, rho, theta, threshold)
        
        lines_list = []
        if lines is not None:
            for i in range(min(len(lines), 100)):  # Limit to first 100 lines
                rho_val, theta_val = lines[i][0]
                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho_val
                y0 = b * rho_val
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                lines_list.append((rho_val, theta_val, x1, y1, x2, y2))
        
        return lines_image, lines_list
    
    def detect_line_segments(self, min_line_length: int = 100, 
                           max_line_gap: int = 10) -> Tuple[np.ndarray, List]:
        """
        Detect line segments using Probabilistic Hough Line Transform.
        
        Args:
            min_line_length: Minimum length of line segments
            max_line_gap: Maximum allowed gap between line segments
            
        Returns:
            Tuple of (image_with_line_segments, segments_list)
        """
        # Create a copy of the original image for drawing
        segments_image = self.original_image.copy()
        
        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            self.edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        segments_list = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(segments_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                segments_list.append((x1, y1, x2, y2))
        
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
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            
            # Check if horizontal (close to 0° or 180°)
            if angle < angle_tolerance or angle > (180 - angle_tolerance):
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(grid_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Check if vertical (close to 90°)
            elif abs(angle - 90) < angle_tolerance:
                vertical_lines.append((x1, y1, x2, y2))
                cv2.line(grid_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Find intersections
        intersections = self._find_intersections(horizontal_lines, vertical_lines)
        
        # Draw intersections
        for x, y in intersections:
            cv2.circle(grid_image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
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
        print("Analyzing image with Hough transforms...")
        
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
        output_dir = f"{base_name}_hough_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results
        cv2.imwrite(f"{output_dir}/circles_detected.png", circles_img)
        cv2.imwrite(f"{output_dir}/lines_detected.png", lines_img)
        cv2.imwrite(f"{output_dir}/segments_detected.png", segments_img)
        cv2.imwrite(f"{output_dir}/grid_detected.png", grid_img)
        cv2.imwrite(f"{output_dir}/edges.png", self.edges)
        
        # Create combined visualization
        self._create_combined_visualization(circles_img, lines_img, segments_img, grid_img, output_dir)
        
        print(f"Results saved to {output_dir}/")
    
    def _create_combined_visualization(self, circles_img: np.ndarray, lines_img: np.ndarray,
                                     segments_img: np.ndarray, grid_img: np.ndarray, 
                                     output_dir: str) -> None:
        """Create a combined visualization of all results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Edges
        axes[0, 1].imshow(self.edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
        
        # Circles
        axes[0, 2].imshow(cv2.cvtColor(circles_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Circle Detection')
        axes[0, 2].axis('off')
        
        # Lines
        axes[1, 0].imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Line Detection')
        axes[1, 0].axis('off')
        
        # Line segments
        axes[1, 1].imshow(cv2.cvtColor(segments_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Line Segments')
        axes[1, 1].axis('off')
        
        # Rectilinear patterns
        axes[1, 2].imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Rectilinear Patterns')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/combined_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the Hough transform analysis."""
    parser = argparse.ArgumentParser(description='Perform Hough transform analysis on PNG images')
    parser.add_argument('image_path', help='Path to the input PNG image')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save result images')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = HoughTransformAnalyzer(args.image_path)
        
        # Perform analysis
        results = analyzer.analyze_all(save_results=not args.no_save)
        
        # Print summary
        print("\n" + "="*50)
        print("HOUGH TRANSFORM ANALYSIS SUMMARY")
        print("="*50)
        print(f"Circles detected: {results['circles']['count']}")
        print(f"Lines detected: {results['lines']['count']}")
        print(f"Line segments detected: {results['segments']['count']}")
        print(f"Horizontal lines: {results['grid']['horizontal_lines']}")
        print(f"Vertical lines: {results['grid']['vertical_lines']}")
        print(f"Grid intersections: {results['grid']['intersections']}")
        print(f"Grid score: {results['grid']['grid_score']:.3f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 