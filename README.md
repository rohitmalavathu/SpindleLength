# **Quantifying Spindle Length in S. pombe**

---

## **Overview**
This project provides a tool for the Hauf Lab at Virginia Tech to quantify spindle length in S. pombe by identifying patterns, calculating principal axes, and measuring distances between structures. It processes multi-frame TIFF images to detect cell structures and measure their dimensions in micrometers.

---

## **Data Sources**
The code processes TIFF image stacks, extracting intensity profiles and measuring distances between detected structures. The analysis is performed on images of S. pombe, with automated detection of patterns and measurement of dimensions. The images were provided by Dr. Silke Hauf.

---

## **Methodology**
1. **Image Pre-processing** - Employs adaptive histogram equalization, denoising, and smoothing to prepare images for analysis
2. **Automated Thresholding** - Uses Otsu's method for initial segmentation followed by morphological operations
3. **Principal Component Analysis** - Calculates the principal axes of identified structures 
4. **Pattern Recognition** - Evaluates correlation scores to identify meaningful patterns in thresholded images
5. **Distance Measurement** - Calculates physical distances between detected endpoints in micrometers

---

## **Key Features**
- **Automated Pattern Detection** - Identifies meaningful structures using correlation analysis
- **Principal Axis Calculation** - Determines the main axis of elongated structures
- **Intensity Profile Analysis** - Samples intensity values along the principal axis
- **Micron Conversion** - Converts pixel measurements to physical distances using TIFF metadata
- **Batch Processing** - Processes multiple TIFF files in a specified folder

---

## **Dependencies**
Ensure the following Python libraries are installed:
- numpy
- matplotlib
- scikit-image
- scipy
- scikit-learn
- tifffile

---

## **Installation**

### **Clone this repository:**
```sh
git clone <repository_url>
cd cell-image-analysis
```

### **Create a virtual environment and activate it:**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **Install dependencies:**
```sh
pip install numpy matplotlib scikit-image scipy scikit-learn tifffile
```

---

## **Usage**

### **Prepare the Dataset:**
- Place your TIFF image stacks in a folder named "Movies"
- The script expects multi-frame TIFF files with embedded resolution metadata

### **Run the Code:**
Execute the script to process all TIFF files in the Movies folder:
```sh
python main.py
```

### **Output:**
- The script creates a "Processed_Results" folder containing:
  - Analysis GIFs showing detected structures in each frame
  - A CSV file with length measurements for each detected structure
- Example:
<img src="01_CPY_analysis.gif" width="600px">

---

## **Function Descriptions**

- `convert_to_microns()`: Extracts resolution metadata from TIFF files to convert pixels to microns
- `find_peak_intersections()`: Identifies intersections between intensity profiles and threshold values
- `sample_intensities_along_axis()`: Samples intensity values along a specified axis
- `find_equal_area_threshold()`: Calculates threshold values based on intensity profiles
- `analyze_threshold_patterns_with_intensities()`: Main analysis function for detecting patterns
- `find_longest_positive_slope()`: Identifies consistent growth trends in measurements
- `analyze_thresholded_points_correlation()`: Evaluates correlation scores for detected patterns
- `find_center_of_mass_and_principal_axis()`: Calculates principal component analysis for structures
- `find_principal_axis_endpoints_in_mask()`: Finds endpoints of structures along principal axes
- `save_lengths_to_csv()`: Exports measurement results to CSV format

---

## **Future Improvements**
- **GUI Interface** - Develop a graphical interface for easier parameter adjustment
- **3D Analysis** - Extend to full 3D analysis of z-stacks
- **Machine Learning Integration** - Add ML-based pattern detection capabilities
- **Enhanced Visualization** - Implement 3D rendering of structures and measurements
- **Automated Reporting** - Generate detailed analysis reports with statistics and visualizations

---

## **Author**
Rohit Malavathu
rohitmalavathu@vt.edu
