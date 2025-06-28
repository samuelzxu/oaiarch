#!/usr/bin/env python3
"""
Hough Transform Comparison Tool

This script runs both OpenCV and scikit-image versions of Hough transform
analysis on the same image and compares the results.

Requirements:
    pip install opencv-python scikit-image

Usage:
    python compare_hough_methods.py input_image.png
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

# Import both analyzers
try:
    from hough_transform import HoughTransformAnalyzer as OpenCVAnalyzer
except ImportError:
    print("Warning: OpenCV version not available. Make sure hough_transform.py exists and opencv-python is installed.")
    OpenCVAnalyzer = None

try:
    from hough_transform_sklearn import HoughTransformAnalyzerSklearn as SklearnAnalyzer
except ImportError:
    print("Warning: Scikit-image version not available. Make sure hough_transform_sklearn.py exists and scikit-image is installed.")
    SklearnAnalyzer = None


def run_analysis(analyzer_class, analyzer_name: str, image_path: str) -> Dict[str, Any]:
    """Run analysis with a specific analyzer and measure performance."""
    print(f"\n{'='*60}")
    print(f"Running {analyzer_name} Analysis")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        analyzer = analyzer_class(image_path)
        results = analyzer.analyze_all(save_results=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results['execution_time'] = execution_time
        results['success'] = True
        
        print(f"\n{analyzer_name} Results:")
        print(f"Circles: {results['circles']['count']}")
        print(f"Lines: {results['lines']['count']}")
        print(f"Line segments: {results['segments']['count']}")
        print(f"Grid score: {results['grid']['grid_score']:.3f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error running {analyzer_name}: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': 0
        }


def compare_results(opencv_results: Dict[str, Any], sklearn_results: Dict[str, Any]) -> None:
    """Compare results from both analyzers."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if not opencv_results['success'] or not sklearn_results['success']:
        print("Cannot compare results - one or both analyses failed.")
        return
    
    # Compare detection counts
    print(f"{'Metric':<20} {'OpenCV':<15} {'Scikit-Image':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics = [
        ('Circles', 'circles', 'count'),
        ('Lines', 'lines', 'count'),
        ('Line Segments', 'segments', 'count'),
        ('Horizontal Lines', 'grid', 'horizontal_lines'),
        ('Vertical Lines', 'grid', 'vertical_lines'),
        ('Grid Intersections', 'grid', 'intersections')
    ]
    
    for metric_name, category, key in metrics:
        opencv_val = opencv_results[category][key]
        sklearn_val = sklearn_results[category][key]
        diff = abs(opencv_val - sklearn_val)
        
        print(f"{metric_name:<20} {opencv_val:<15} {sklearn_val:<15} {diff:<15}")
    
    # Compare grid scores
    opencv_score = opencv_results['grid']['grid_score']
    sklearn_score = sklearn_results['grid']['grid_score']
    score_diff = abs(opencv_score - sklearn_score)
    
    print(f"{'Grid Score':<20} {opencv_score:<15.3f} {sklearn_score:<15.3f} {score_diff:<15.3f}")
    
    # Compare execution times
    opencv_time = opencv_results['execution_time']
    sklearn_time = sklearn_results['execution_time']
    time_diff = abs(opencv_time - sklearn_time)
    
    print(f"{'Execution Time (s)':<20} {opencv_time:<15.2f} {sklearn_time:<15.2f} {time_diff:<15.2f}")
    
    # Performance comparison
    print(f"\nPerformance Analysis:")
    if opencv_time < sklearn_time:
        speedup = sklearn_time / opencv_time
        print(f"OpenCV is {speedup:.2f}x faster than scikit-image")
    elif sklearn_time < opencv_time:
        speedup = opencv_time / sklearn_time
        print(f"Scikit-image is {speedup:.2f}x faster than OpenCV")
    else:
        print("Both methods have similar execution times")
    
    # Detection quality comparison
    print(f"\nDetection Quality Analysis:")
    total_opencv = sum([opencv_results[cat]['count'] for cat in ['circles', 'lines', 'segments']])
    total_sklearn = sum([sklearn_results[cat]['count'] for cat in ['circles', 'lines', 'segments']])
    
    if total_opencv > total_sklearn:
        print(f"OpenCV detected {total_opencv - total_sklearn} more features overall")
    elif total_sklearn > total_opencv:
        print(f"Scikit-image detected {total_sklearn - total_opencv} more features overall")
    else:
        print("Both methods detected the same number of features overall")


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description='Compare OpenCV and scikit-image Hough transform implementations')
    parser.add_argument('image_path', help='Path to the input PNG image')
    parser.add_argument('--opencv-only', action='store_true', help='Run only OpenCV analysis')
    parser.add_argument('--sklearn-only', action='store_true', help='Run only scikit-image analysis')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return 1
    
    # Check which analyzers are available
    if args.opencv_only and OpenCVAnalyzer is None:
        print("Error: OpenCV analyzer not available.")
        return 1
    
    if args.sklearn_only and SklearnAnalyzer is None:
        print("Error: Scikit-image analyzer not available.")
        return 1
    
    if not args.opencv_only and not args.sklearn_only:
        if OpenCVAnalyzer is None and SklearnAnalyzer is None:
            print("Error: No analyzers available.")
            return 1
    
    print(f"Analyzing image: {args.image_path}")
    
    opencv_results = None
    sklearn_results = None
    
    # Run OpenCV analysis
    if not args.sklearn_only and OpenCVAnalyzer is not None:
        opencv_results = run_analysis(OpenCVAnalyzer, "OpenCV", args.image_path)
    
    # Run scikit-image analysis
    if not args.opencv_only and SklearnAnalyzer is not None:
        sklearn_results = run_analysis(SklearnAnalyzer, "Scikit-Image", args.image_path)
    
    # Compare results if both were run
    if opencv_results is not None and sklearn_results is not None:
        compare_results(opencv_results, sklearn_results)
    
    print(f"\nAnalysis complete! Check the output directories for visual results.")
    
    return 0


if __name__ == "__main__":
    exit(main()) 