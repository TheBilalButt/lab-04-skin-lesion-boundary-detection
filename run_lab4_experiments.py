"""
Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection
Course: Computer Vision
Author: Bilal Butt (@TheBilalButt)
Dataset: ISIC Skin-Lesion Diagnostic Dataset (Melanoma, Nevus, BCC, BKL, DF)

This script implements the full end-to-end experimental pipeline:
  - Task 1: Image Acquisition & Ingestion
  - Task 2: Grayscale & Gaussian Smoothing Preprocessing
  - Task 3: Multi-Regime Canny Edge Detection (50-100, 100-200, 150-250)
  - Task 4: Comparative Threshold Analysis & Selection
  - Task 5: Morphological Lesion Boundary Extraction & Overlay
  - Task 6: Quantitative Morphometrics (Area & Perimeter)
  - Required Pipeline Visualization (Original -> Gray -> Gaussian -> Canny -> Boundary)
  - Filter x Edge Detector Benchmarking (8 Combinations)
"""

import os
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure output directories exist
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Image metadata mapping
IMAGE_INFO = [
    {
        "id": "Image 1",
        "file": "image_1.jpg",
        "diagnosis": "Melanoma (MEL)",
        "type": "Malignant Melanocytic Neoplasm",
        "isic_id": "ISIC_0000029"
    },
    {
        "id": "Image 2",
        "file": "image_2.jpg",
        "diagnosis": "Melanocytic Nevus (NV)",
        "type": "Benign Melanocytic Lesion",
        "isic_id": "ISIC_0000039"
    },
    {
        "id": "Image 3",
        "file": "image_3.jpg",
        "diagnosis": "Basal Cell Carcinoma (BCC)",
        "type": "Malignant Non-Melanocytic Lesion",
        "isic_id": "ISIC_0024665"
    },
    {
        "id": "Image 4",
        "file": "image_4.jpg",
        "diagnosis": "Seborrheic Keratosis (BKL)",
        "type": "Benign Keratinocytic Lesion",
        "isic_id": "ISIC_0010862"
    },
    {
        "id": "Image 5",
        "file": "image_5.jpg",
        "diagnosis": "Dermatofibroma (DF)",
        "type": "Benign Fibrohistiocytic Lesion",
        "isic_id": "ISIC_0024973"
    }
]

def load_dataset():
    """Load all 5 standardized dermoscopic images."""
    dataset = []
    for info in IMAGE_INFO:
        path = os.path.join('images', info['file'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required image: {path}")
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dataset.append({
            **info,
            "path": path,
            "bgr": img_bgr,
            "rgb": img_rgb,
            "shape": img_rgb.shape
        })
    return dataset

# ==============================================================================
# Preprocessing & Edge Detection Primitives
# ==============================================================================

def preprocess(img_bgr, ksize=(5, 5), sigma=1.5):
    """Task 2: Convert to grayscale and apply Gaussian filter."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, ksize, sigma)
    return gray, gauss

def apply_canny_regimes(gauss_img):
    """Task 3: Apply Canny with three standard threshold settings."""
    regimes = {
        "50-100": cv2.Canny(gauss_img, 50, 100),
        "100-200": cv2.Canny(gauss_img, 100, 200),
        "150-250": cv2.Canny(gauss_img, 150, 250)
    }
    return regimes

def extract_lesion_boundary(edge_map, img_bgr, margin=15, kernel_size=(11, 11), iterations=2):
    """
    Task 5: Extract contiguous lesion boundary from edges using morphological
    closing and external contour detection.
    """
    h, w = edge_map.shape
    edges_clean = edge_map.copy()
    
    # Suppress outer canvas border to avoid camera/frame boundary artifacts
    edges_clean[:margin, :] = 0
    edges_clean[-margin:, :] = 0
    edges_clean[:, :margin] = 0
    edges_clean[:, -margin:] = 0
    
    # Morphological closing to bridge micro-fractures in the Canny edge perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(edges_clean, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter candidates by area to discard isolated noise speckles
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    if not valid_contours and contours:
        best_contour = max(contours, key=cv2.contourArea)
    elif valid_contours:
        best_contour = max(valid_contours, key=cv2.contourArea)
    else:
        best_contour = None

    vis_bgr = img_bgr.copy()
    area = 0.0
    perimeter = 0.0

    if best_contour is not None:
        area = float(cv2.contourArea(best_contour))
        perimeter = float(cv2.arcLength(best_contour, closed=True))
        
        # Transparent overlay fill
        overlay = vis_bgr.copy()
        cv2.drawContours(overlay, [best_contour], -1, (0, 255, 120), -1)
        cv2.addWeighted(overlay, 0.25, vis_bgr, 0.75, 0, vis_bgr)
        
        # Draw high-visibility boundary contour line
        cv2.drawContours(vis_bgr, [best_contour], -1, (0, 255, 0), 2)
        
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return best_contour, area, perimeter, vis_rgb, closed

# ==============================================================================
# Visualization Generators
# ==============================================================================

def generate_task1_figure(dataset):
    """Plot original images with diagnostic taxonomy."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, data in enumerate(dataset):
        axes[i].imshow(data['rgb'])
        axes[i].set_title(f"{data['id']}\n{data['diagnosis']}\n({data['isic_id']})", fontsize=10, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    out_path = os.path.join('figures', 'task1_original_images.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_task2_figure(dataset):
    """Plot Original vs Grayscale vs Gaussian Filtered."""
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    for i, data in enumerate(dataset):
        gray, gauss = preprocess(data['bgr'])
        
        # Column 1: Original
        axes[i, 0].imshow(data['rgb'])
        axes[i, 0].set_title(f"{data['id']} - Original", fontsize=10)
        axes[i, 0].axis('off')
        
        # Column 2: Grayscale
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title(f"{data['id']} - Grayscale", fontsize=10)
        axes[i, 1].axis('off')
        
        # Column 3: Gaussian Filtered
        axes[i, 2].imshow(gauss, cmap='gray')
        axes[i, 2].set_title(f"{data['id']} - Gaussian Filter (5x5, \u03c3=1.5)", fontsize=10)
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join('figures', 'task2_preprocessing.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_task3_figure(dataset):
    """Plot Canny Edge Detection across 3 threshold regimes."""
    fig, axes = plt.subplots(5, 3, figsize=(13, 17))
    for i, data in enumerate(dataset):
        _, gauss = preprocess(data['bgr'])
        regimes = apply_canny_regimes(gauss)
        
        for j, (label, edge_map) in enumerate(regimes.items()):
            n_edges = np.count_nonzero(edge_map)
            axes[i, j].imshow(edge_map, cmap='gray')
            axes[i, j].set_title(f"{data['id']} | Canny [{label}]\nEdge Pixels: {n_edges}", fontsize=9)
            axes[i, j].axis('off')
            
    plt.tight_layout()
    out_path = os.path.join('figures', 'task3_canny_thresholds.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_task4_figure(dataset):
    """Threshold sensitivity comparison showing edge degradation at high thresholds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds = ["50-100", "100-200", "150-250"]
    
    for data in dataset:
        _, gauss = preprocess(data['bgr'])
        regimes = apply_canny_regimes(gauss)
        counts = [np.count_nonzero(regimes[t]) for t in thresholds]
        ax.plot(thresholds, counts, marker='o', linewidth=2, label=f"{data['id']} ({data['diagnosis']})")
        
    ax.set_title("Canny Edge Pixel Retention across Threshold Regimes", fontsize=12, fontweight='bold')
    ax.set_xlabel("Threshold Setting (Low - High)", fontsize=11)
    ax.set_ylabel("Total Detected Edge Pixels", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join('figures', 'task4_best_canny_selection.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_task5_figure(dataset):
    """Plot detected lesion boundary overlaid on original color image."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, data in enumerate(dataset):
        _, gauss = preprocess(data['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, vis_rgb, _ = extract_lesion_boundary(edges, data['bgr'])
        
        axes[i].imshow(vis_rgb)
        axes[i].set_title(f"{data['id']} Boundary\nArea: {area:,.0f} px\nPerim: {perim:,.1f} px", fontsize=9, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    out_path = os.path.join('figures', 'task5_lesion_boundary.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_required_pipeline_strip(dataset):
    """
    Required Visualization:
    Original -> Grayscale -> Gaussian Filter -> Canny -> Lesion Boundary
    """
    fig, axes = plt.subplots(5, 5, figsize=(18, 16))
    headers = ["Original Image", "Grayscale", "Gaussian Filter", "Canny Edges (50-100)", "Lesion Boundary"]
    
    for i, data in enumerate(dataset):
        gray, gauss = preprocess(data['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, vis_rgb, _ = extract_lesion_boundary(edges, data['bgr'])
        
        imgs = [data['rgb'], gray, gauss, edges, vis_rgb]
        cmaps = [None, 'gray', 'gray', 'gray', None]
        
        for col_idx in range(5):
            axes[i, col_idx].imshow(imgs[col_idx], cmap=cmaps[col_idx])
            if i == 0:
                axes[i, col_idx].set_title(headers[col_idx], fontsize=11, fontweight='bold', pad=10)
            if col_idx == 0:
                axes[i, col_idx].set_ylabel(f"{data['id']}\n{data['diagnosis']}", fontsize=10, fontweight='bold')
            if col_idx == 4:
                axes[i, col_idx].set_xlabel(f"A={area:,.0f} | P={perim:,.1f} px", fontsize=9, labelpad=5)
            axes[i, col_idx].set_xticks([])
            axes[i, col_idx].set_yticks([])
            
    plt.tight_layout()
    out_path = os.path.join('figures', 'required_visualization_pipeline.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_comparison_grid(sample_data):
    """
    Generate 2x4 comparison figure of the 8 filter x edge detector combinations
    on representative sample (Image 1 - Melanoma).
    """
    img_bgr = sample_data['bgr']
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Filter variants
    filters = {
        "Original": gray,
        "Average": cv2.blur(gray, (5, 5)),
        "Gaussian": cv2.GaussianBlur(gray, (5, 5), 1.5),
        "Median": cv2.medianBlur(gray, 5)
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    methods = [
        ("Original + Sobel", filters["Original"], "sobel", (0, 0)),
        ("Original + Canny", filters["Original"], "canny", (0, 1)),
        ("Average + Sobel", filters["Average"], "sobel", (0, 2)),
        ("Average + Canny", filters["Average"], "canny", (0, 3)),
        ("Gaussian + Sobel", filters["Gaussian"], "sobel", (1, 0)),
        ("Gaussian + Canny", filters["Gaussian"], "canny", (1, 1)),
        ("Median + Sobel", filters["Median"], "sobel", (1, 2)),
        ("Median + Canny", filters["Median"], "canny", (1, 3))
    ]
    
    for title, src_img, detector, (r, c) in methods:
        if detector == "sobel":
            sx = cv2.Sobel(src_img, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(src_img, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(sx, sy)
            mag = np.uint8(np.clip(mag / mag.max() * 255.0, 0, 255))
            # threshold sobel magnitude
            _, edge_vis = cv2.threshold(mag, 45, 255, cv2.THRESH_BINARY)
        else:
            edge_vis = cv2.Canny(src_img, 50, 100)
            
        axes[r, c].imshow(edge_vis, cmap='gray')
        axes[r, c].set_title(title, fontsize=11, fontweight='bold')
        axes[r, c].axis('off')
        
    plt.suptitle(f"Comprehensive Filter x Edge Detector Benchmark ({sample_data['diagnosis']})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join('figures', 'final_comparison_filters_edges.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

# ==============================================================================
# Quantitative Tabulation & Reporting
# ==============================================================================

def compute_and_save_tables(dataset):
    """Compute measurements table and export final comparison benchmarks."""
    results_list = []
    
    print("\n" + "="*80)
    print("TABLE 1: LESION AREA AND PERIMETER MEASUREMENTS")
    print("="*80)
    print(f"{'Image':<10} | {'Best Filter':<24} | {'Edge Method':<14} | {'Area (pixels)':<15} | {'Perimeter (pixels)':<18}")
    print("-" * 88)
    
    for data in dataset:
        _, gauss = preprocess(data['bgr'])
        edges = cv2.Canny(gauss, 50, 100)
        _, area, perim, _, _ = extract_lesion_boundary(edges, data['bgr'])
        
        row = {
            "Image": data['id'],
            "Diagnosis": data['diagnosis'],
            "ISIC_ID": data['isic_id'],
            "Best Filter": "Gaussian (5x5, sigma=1.5)",
            "Edge Method": "Canny (50-100)",
            "Area (pixels)": round(area, 1),
            "Perimeter (pixels)": round(perim, 1),
            "Circularity": round((4 * np.pi * area) / (perim**2 + 1e-6), 3) if perim > 0 else 0.0
        }
        results_list.append(row)
        print(f"{row['Image']:<10} | {row['Best Filter']:<24} | {row['Edge Method']:<14} | {row['Area (pixels)']:<15,.1f} | {row['Perimeter (pixels)']:<18,.1f}")

    # Save to CSV
    csv_path = os.path.join('results', 'lesion_measurements.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results_list[0].keys()))
        writer.writeheader()
        writer.writerows(results_list)
    print(f"\n[+] Saved measurements table to: {csv_path}")

    # Save to JSON
    json_path = os.path.join('results', 'lesion_measurements.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2)
    print(f"[+] Saved measurements JSON to: {json_path}")

    # Final Comparison Benchmark Table (8 methods)
    comparison_benchmarks = [
        {
            "Method": "Original + Sobel",
            "Noise Handling": "Poor (High sensitivity to hair, pores & texture)",
            "Edge Quality": "Thick, diffuse edges with severe noise clutter",
            "Boundary Detection": "Unreliable; perimeter drowned in background skin noise",
            "Overall Performance": "Poor"
        },
        {
            "Method": "Original + Canny",
            "Noise Handling": "Fair (Hysteresis suppresses isolated weak noise)",
            "Edge Quality": "Thin (1-pixel) but cluttered with micro-vessel artifacts",
            "Boundary Detection": "Fragmented; noisy false edges bridge into border",
            "Overall Performance": "Moderate"
        },
        {
            "Method": "Average + Sobel",
            "Noise Handling": "Moderate (Uniform spatial smoothing of speckle)",
            "Edge Quality": "Blurred transitions, significant edge displacement",
            "Boundary Detection": "Imprecise; low contrast boundaries are washed out",
            "Overall Performance": "Fair"
        },
        {
            "Method": "Average + Canny",
            "Noise Handling": "Good (Box blur suppresses high-frequency noise)",
            "Edge Quality": "Sharp edges, slight boundary position drift",
            "Boundary Detection": "Good; captures major contour with minor gaps",
            "Overall Performance": "Good"
        },
        {
            "Method": "Gaussian + Sobel",
            "Noise Handling": "Good (Isotropic frequency damping preserves gradients)",
            "Edge Quality": "Smooth continuous response, but gradient edges remain thick",
            "Boundary Detection": "Fair to Good; requires empirical threshold tuning",
            "Overall Performance": "Good"
        },
        {
            "Method": "Gaussian + Canny",
            "Noise Handling": "Optimal (Gaussian smoothing + dual-threshold hysteresis)",
            "Edge Quality": "Excellent; strictly thin, non-maximum suppressed edges",
            "Boundary Detection": "Superior; coherent, well-localized closed boundary",
            "Overall Performance": "Best (Benchmark)"
        },
        {
            "Method": "Median + Sobel",
            "Noise Handling": "Very Good (Non-linear rejection of salt-and-pepper / hair)",
            "Edge Quality": "Sharp step-edge preservation without gradient smearing",
            "Boundary Detection": "Good; robust against hair and epidermal pore noise",
            "Overall Performance": "Very Good"
        },
        {
            "Method": "Median + Canny",
            "Noise Handling": "Excellent (Exceptional hair/specular artifact rejection)",
            "Edge Quality": "Thin, crisp edges with minimal spurious responses",
            "Boundary Detection": "Excellent; highly accurate boundary delineation",
            "Overall Performance": "Excellent (Alternative)"
        }
    ]

    print("\n" + "="*110)
    print("TABLE 2: FINAL COMPARISON ACROSS FILTER AND EDGE DETECTION METHODS")
    print("="*110)
    print(f"{'Method':<18} | {'Noise Handling':<25} | {'Edge Quality':<25} | {'Boundary Detection':<25} | {'Overall'}")
    print("-" * 110)
    for row in comparison_benchmarks:
        print(f"{row['Method']:<18} | {row['Noise Handling'][:24]:<25} | {row['Edge Quality'][:24]:<25} | {row['Boundary Detection'][:24]:<25} | {row['Overall Performance']}")

    comp_csv_path = os.path.join('results', 'filter_edge_comparison.csv')
    with open(comp_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_benchmarks[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_benchmarks)
    print(f"\n[+] Saved filter comparison table to: {comp_csv_path}\n")

def main():
    print("[*] Starting Computer Vision Lab 4 Experiments...")
    print("[*] Loading dermoscopic image dataset...")
    dataset = load_dataset()
    print(f"[+] Loaded {len(dataset)} dermoscopic benchmark images.")
    
    print("[*] Executing Task 1: Image Visualization...")
    generate_task1_figure(dataset)
    
    print("[*] Executing Task 2: Grayscale & Gaussian Smoothing...")
    generate_task2_figure(dataset)
    
    print("[*] Executing Task 3: Canny Threshold Parameter Study...")
    generate_task3_figure(dataset)
    
    print("[*] Executing Task 4: Selecting Optimal Threshold...")
    generate_task4_figure(dataset)
    
    print("[*] Executing Task 5: Morphological Boundary Delineation...")
    generate_task5_figure(dataset)
    
    print("[*] Executing Required Pipeline Strip (Original -> Gray -> Gauss -> Canny -> Boundary)...")
    generate_required_pipeline_strip(dataset)
    
    print("[*] Executing Final 8-Method Comparison Grid...")
    generate_comparison_grid(dataset[0])
    
    print("[*] Computing Quantitative Morphometrics & Summary Tables...")
    compute_and_save_tables(dataset)
    
    print("[+] All Lab 4 experiments completed successfully.")

if __name__ == '__main__':
    main()
