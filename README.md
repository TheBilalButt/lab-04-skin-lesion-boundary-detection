# Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheBilalButt/lab-04-skin-lesion-boundary-detection/blob/main/Lab4_Skin_Lesion_Boundary_Detection.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/TheBilalButt/lab-04-skin-lesion-boundary-detection)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?logo=opencv)](https://opencv.org/)

**Course**: Computer Vision / Medical Image Processing  
**Author**: Bilal Butt ([@TheBilalButt](https://github.com/TheBilalButt))  
**Dataset**: International Skin Imaging Collaboration (ISIC) Benchmark Dataset  
**Repository**: [lab-04-skin-lesion-boundary-detection](https://github.com/TheBilalButt/lab-04-skin-lesion-boundary-detection)

---

## 1. Problem Statement

Develop a simple computer vision system that detects the boundary of a skin lesion from a skin image using image filtering and Canny edge detection.  
The primary clinical and algorithmic goal is to determine how effectively classical edge detection can separate the lesion from surrounding healthy skin tissue, overcome dermal noise artifacts, and quantify diagnostic morphometric features (area and perimeter).

---

## 2. Methodology & System Architecture

The boundary delineation pipeline follows a multi-stage deterministic architecture:

```
[Input RGB Skin Image]
         │
         ▼
[Grayscale Conversion]  ──>  Scalar luminance intensity mapping (Y = 0.299R + 0.587G + 0.114B)
         │
         ▼
[Gaussian Smoothing]   ──>  Isotropic 2D kernel (5x5, σ = 1.5) attenuates epidermal texture/pores
         │
         ▼
[Canny Edge Detection] ──>  Gradient Sobel filtering, non-maximum suppression (1-px edges), dual-threshold hysteresis
         │
         ▼
[Morphological Closing]──>  Elliptical kernel structuring element bridges perimeter micro-fractures
         │
         ▼
[Contour Detection]    ──>  External topological contour extraction, artifact suppression & area/perimeter morphometrics
         │
         ▼
[Visual Boundary Map]  ──>  Accurate perimeter overlay on the original diagnostic color image
```

---

## 3. Experimental Tasks & Results

### Task 1 — Load the Image
Five standardized dermoscopic images were curated from the ISIC diagnostic repository spanning malignant and benign lesions:
- **Image 1**: Melanoma (MEL) — `ISIC_0000029`
- **Image 2**: Melanocytic Nevus (NV) — `ISIC_0000039`
- **Image 3**: Basal Cell Carcinoma (BCC) — `ISIC_0024665`
- **Image 4**: Seborrheic Keratosis (BKL) — `ISIC_0010862`
- **Image 5**: Dermatofibroma (DF) — `ISIC_0024973`

![Task 1 - Original Images](figures/task1_original_images.png)

---

### Task 2 — Preprocess the Image
Dermoscopy images exhibit severe high-frequency interference arising from epidermal furrows, fine hairs, skin pores, and specular camera reflections.
1. Images were converted to scalar grayscale to eliminate chromatic variance while preserving pigment luminance transitions.
2. A normalized 2D Gaussian filter kernel ($5 \times 5$, $\sigma = 1.5$) was applied to regularize spatial gradients:
   $$G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

![Task 2 - Preprocessing](figures/task2_preprocessing.png)

---

### Task 3 — Apply Canny Edge Detection
Canny edge detection was applied across three distinct dual-threshold configurations:
- **Low Thresholds (50–100)**: Captures subtle lesion perimeter transitions; sensitive to micro-textures.
- **Medium Thresholds (100–200)**: Standard operating threshold; detects high-contrast edges.
- **High Thresholds (150–250)**: Conservative threshold; retains only extreme step-edge transitions.

![Task 3 - Canny Threshold Regimes](figures/task3_canny_thresholds.png)

---

### Task 4 — Select the Best Result

#### Comparative Edge Retention Analysis
The number of edge pixels retained across threshold regimes demonstrates steep drop-offs:

![Task 4 - Threshold Selection](figures/task4_best_canny_selection.png)

#### Technical Rationale for Threshold Selection
1. **High Thresholds (150–250) Fail on Organic Boundaries**:
   Dermoscopic transitions between healthy epidermis and pigmented lesions are biological, continuous gradient transitions rather than synthetic sharp step edges. Gaussian pre-smoothing ($\sigma=1.5$) lowers gradient magnitudes below 150 across most of the perimeter. Setting $T_{\text{high}} = 250$ completely suppresses edge formation, yielding near-zero detected edges.
2. **Medium Thresholds (100–200) Cause Perimeter Fragmentation**:
   While central high-contrast nodules trigger edges, peripheral sections where pigmentation gradually diffuses fail to reach $T_{\text{low}} = 100$. Hysteresis tracking stops prematurely, leaving disconnected edge fragments.
3. **Low Thresholds (50–100) Provide Optimal Continuity**:
   Setting $T_{\text{low}} = 50$ allows the hysteresis algorithm to track weak edges connected to definite edge anchors ($T_{\text{high}} = 100$). Because Gaussian pre-filtering has already suppressed high-frequency noise, this setting yields an unbroken, continuous outline around the entire lesion perimeter.

---

### Task 5 — Detect the Lesion Boundary
To extract a closed mathematical contour from Canny edges:
1. **Margin Clamping**: A 15-pixel border margin is zeroed out to prevent camera lens vignetting and canvas borders from triggering false contours.
2. **Morphological Closing**: A 2D elliptical structuring element ($11 \times 11$, 2 iterations) bridges edge gaps:
   $$A \bullet B = (A \oplus B) \ominus B$$
3. **External Topological Contour Extraction**: `cv2.findContours(..., cv2.RETR_EXTERNAL)` isolates the dominant lesion envelope.
4. **Boundary Overlay**: The primary contour is rendered in high-contrast green ($thickness=2$) with a semi-transparent diagnostic fill ($alpha=0.25$).

![Task 5 - Lesion Boundary](figures/task5_lesion_boundary.png)

---

### Task 6 — Calculate Lesion Area & Required Visualization

#### Required Visualization Pipeline Strip
`Original Image → Grayscale → Gaussian Filter → Canny → Lesion Boundary`

![Required Visualization Pipeline](figures/required_visualization_pipeline.png)

#### Quantitative Morphometric Table
Below are the measured area and perimeter metrics for all 5 test lesions:

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| :--- | :--- | :--- | :--- | :--- |
| **Image 1** | Gaussian (5x5, sigma=1.5) | Canny (50-100) | 5,289.5 | 1,722.0 |
| **Image 2** | Gaussian (5x5, sigma=1.5) | Canny (50-100) | 1,979.0 | 508.8 |
| **Image 3** | Gaussian (5x5, sigma=1.5) | Canny (50-100) | 2,445.5 | 630.6 |
| **Image 4** | Gaussian (5x5, sigma=1.5) | Canny (50-100) | 9,656.5 | 494.7 |
| **Image 5** | Gaussian (5x5, sigma=1.5) | Canny (50-100) | 3,837.5 | 527.7 |

---

## 4. Final Comparison: Filter $\times$ Edge Detection Benchmark

To thoroughly evaluate filtering and edge detection synergy, 8 standard combinations were benchmarked on dermoscopic lesion imagery:

![Final Comparison](figures/final_comparison_filters_edges.png)

### Benchmark Evaluation Table

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Original + Sobel** | Poor (High sensitivity to hair, pores & texture) | Thick, diffuse edges with severe noise clutter | Unreliable; perimeter drowned in background skin noise | **Poor** |
| **Original + Canny** | Fair (Hysteresis suppresses isolated weak noise) | Thin (1-pixel) but cluttered with micro-vessel artifacts | Fragmented; noisy false edges bridge into border | **Moderate** |
| **Average + Sobel** | Moderate (Uniform spatial smoothing of speckle) | Blurred transitions, significant edge displacement | Imprecise; low contrast boundaries are washed out | **Fair** |
| **Average + Canny** | Good (Box blur suppresses high-frequency noise) | Sharp edges, slight boundary position drift | Good; captures major contour with minor gaps | **Good** |
| **Gaussian + Sobel** | Good (Isotropic frequency damping preserves gradients) | Smooth continuous response, but gradient edges remain thick | Fair to Good; requires empirical threshold tuning | **Good** |
| **Gaussian + Canny** | Optimal (Gaussian smoothing + dual-threshold hysteresis) | Excellent; strictly thin, non-maximum suppressed edges | Superior; coherent, well-localized closed boundary | **Best (Benchmark)** |
| **Median + Sobel** | Very Good (Non-linear rejection of salt-and-pepper / hair) | Sharp step-edge preservation without gradient smearing | Good; robust against hair and epidermal pore noise | **Very Good** |
| **Median + Canny** | Excellent (Exceptional hair/specular artifact rejection) | Thin, crisp edges with minimal spurious responses | Excellent; highly accurate boundary delineation | **Excellent (Alternative)** |

---

## 5. Answers to Lab Assignment Questions

### 1. Why is Gaussian filtering applied before Canny detection?
Canny edge detection computes spatial gradient approximations using differential operators ($G_x = \partial I / \partial x$, $G_y = \partial I / \partial y$). Differentiation in the spatial domain acts as a high-pass filter that significantly amplifies high-frequency image energy. In skin lesion imagery, high-frequency components are dominated by non-lesion artifacts: epidermal skin furrows, hair shafts, skin pores, and camera sensor noise. Convolving the image with an isotropic 2D Gaussian kernel acts as a low-pass filter, attenuating these rapid intensity fluctuations and ensuring that calculated directional derivatives correspond to genuine macroscopic lesion-skin transitions.

### 2. How did the three Canny threshold settings affect the result?
- **50–100 (Low Setting)**: High sensitivity. Successfully captures the complete boundary contour, even where pigmentation gradually fades into normal skin. Minor skin texture is occasionally retained, but easily bridged or eliminated by subsequent morphology.
- **100–200 (Medium Setting)**: Moderate sensitivity. Retains primary high-contrast edges but fails along diffuse, subtle margins, causing the perimeter to fragment into disconnected line segments.
- **150–250 (High Setting)**: Severe under-detection. Gradient magnitudes in smoothed biological skin tissue rarely exceed 150–200. Consequently, almost no pixels satisfy the high threshold, causing the edge detector to fail completely.

### 3. Which threshold produced the best lesion boundary?
The **50–100** threshold setting produced the best lesion boundary across all five test cases. Combined with Gaussian smoothing ($\sigma=1.5$), it maintained unbroken boundary connectivity without excessive spurious edge noise.

### 4. Why are edges useful for detecting skin lesions?
In clinical dermatology, skin neoplasms (particularly malignant melanoma) are defined by localized hyper-melanocytic proliferation that produces sharp optical contrast against background Caucasian or Fitzpatrick skin tones. Under the **ABCDE clinical diagnostic protocol** (Asymmetry, Border irregularity, Color variegation, Diameter, Evolution), the **Border (B)** is a decisive prognostic indicator:
- **Lesion Area**: Directly quantifies planar tumor growth and horizontal extent.
- **Lesion Perimeter & Circularity**: Quantifies border roughness, notched borders, and radial growth spikes, providing algorithmic indicators for malignancy classification.

### 5. What problems did you observe in detecting the lesion boundary?
1. **Diffuse & Gradual Transitions**: Biological lesions often taper off smoothly without a sharp step edge, making fixed-threshold gradient detection ambiguous.
2. **Hair & Texture Interference**: Fine hair shafts and epidermal ridges create strong gradient spikes that can falsely bridge into the lesion perimeter.
3. **Internal Lesion Variegation**: Multi-colored lesion interiors (melanin globules, regression patches, blood lakes) produce dense internal edges that must be differentiated from the outer boundary.
4. **Peripheral Lens Vignetting**: Darkening at the corners of dermoscopy lenses can trigger spurious peripheral edge responses.

### 6. How could your method be improved?
1. **Perceptual Color Space Gradients**: Compute directional gradients in CIE $L^*a^*b^*$ or HSV color spaces (specifically using the $a^*$ channel for erythema and saturation for melanin) rather than scalar luminance grayscale.
2. **DullRazor Hair Removal**: Prepend a generalized morphological closing with multi-directional line structuring elements to digitally erase hairs before computing gradients.
3. **Dynamic / Adaptive Canny Thresholding**: Compute image-specific thresholds based on Otsu's method or median gradient statistics to adapt to varying skin phototypes.
4. **Variational Active Contours (Snakes / Chan-Vese)**: Use Canny edge maps as an external energy constraint to guide an elastic active contour model that enforces smoothness and topological closure.
5. **Deep Learning Fusion**: Combine edge prior maps with deep semantic segmentation networks (e.g., U-Net with boundary-aware attention) for robust clinical segmentation.

---

## 6. Repository Structure

```
├── .gitignore                          # Clean Git exclusion rules
├── Lab4_Skin_Lesion_Boundary_Detection.ipynb # Interactive Jupyter Notebook
├── README.md                           # Comprehensive documentation & results
├── requirements.txt                    # Python environment specifications
├── run_lab4_experiments.py             # Reproducible end-to-end experiment pipeline
├── generate_notebook.py                # Automated Jupyter Notebook generator
├── generate_readme.py                  # Documentation generator script
├── images/                             # Standardized ISIC dermoscopy benchmark images
│   ├── image_1.jpg                     # Melanoma (MEL)
│   ├── image_2.jpg                     # Melanocytic Nevus (NV)
│   ├── image_3.jpg                     # Basal Cell Carcinoma (BCC)
│   ├── image_4.jpg                     # Seborrheic Keratosis (BKL)
│   └── image_5.jpg                     # Dermatofibroma (DF)
├── figures/                            # High-resolution experiment visualizations
│   ├── task1_original_images.png       # Task 1: Diagnostic input images
│   ├── task2_preprocessing.png         # Task 2: Grayscale & Gaussian filtering
│   ├── task3_canny_thresholds.png      # Task 3: Multi-threshold Canny comparison
│   ├── task4_best_canny_selection.png  # Task 4: Quantitative threshold curve
│   ├── task5_lesion_boundary.png       # Task 5: Detected boundary overlays
│   ├── required_visualization_pipeline.png # 5-Stage required pipeline strip
│   └── final_comparison_filters_edges.png # 8-Method comparison grid
└── results/                            # Tabular results & metrics
    ├── lesion_measurements.csv         # Area and perimeter measurements
    ├── lesion_measurements.json        # Structured JSON measurements
    └── filter_edge_comparison.csv      # 8-Method benchmark evaluation
```

---

## 7. How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/TheBilalButt/lab-04-skin-lesion-boundary-detection.git
cd lab-04-skin-lesion-boundary-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
```bash
python run_lab4_experiments.py
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook Lab4_Skin_Lesion_Boundary_Detection.ipynb
```

---

*Authored by Bilal Butt (@TheBilalButt) for Computer Vision Lab Assignment.*
