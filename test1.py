import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure, measure
import scipy.ndimage as ndimage
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
import tifffile
#import imageio
import glob
import os

def convert_to_microns(file_path):
    with tifffile.TiffFile(file_path) as tif:
        selected_page = tif.pages[0]
        
        x_res_tag = selected_page.tags.get("XResolution")
            
        x_res = x_res_tag.value
        pixel_width = x_res[1] / x_res[0]
    
    return pixel_width
    
def find_peak_intersections(x, y, threshold_value=None, neighbor_window=3):
    import numpy as np
    
    x_arr = np.array(x, dtype=float)    
    y_arr = np.array(y, dtype=float)
    
    if len(x_arr) < 3:
        return [], None, None
    
    smoothed_y = np.array(y_arr, dtype=float)
    for i in range(len(y_arr)):
        start_idx = max(0, i - neighbor_window)
        end_idx = min(len(y_arr), i + neighbor_window + 1)
        smoothed_y[i] = np.mean(y_arr[start_idx:end_idx])
    
    peak_idx = np.argmax(smoothed_y)
    peak_x = x_arr[peak_idx]
    
    if threshold_value is None:
        mean_value = np.mean(smoothed_y)
        std_dev = np.std(smoothed_y)
        threshold_value = float(mean_value + std_dev)
    else:
        threshold_value = float(threshold_value)
    
    left_intersections = []
    for i in range(peak_idx - 1, 0, -1):
        y_curr = float(smoothed_y[i])
        y_prev = float(smoothed_y[i-1])
        
        if (y_curr <= threshold_value <= y_prev) or (y_curr >= threshold_value >= y_prev):
            denominator = y_prev - y_curr
            
            if abs(denominator) > 1e-10:
                try:
                    t = (threshold_value - y_curr) / denominator
                    x_intersect = x_arr[i] + t * (x_arr[i-1] - x_arr[i])
                    left_intersections.append((x_intersect, threshold_value))
                except (OverflowError, FloatingPointError, ZeroDivisionError):
                    x_intersect = (x_arr[i] + x_arr[i-1]) / 2
                    left_intersections.append((x_intersect, threshold_value))
            else:
                x_intersect = (x_arr[i] + x_arr[i-1]) / 2
                left_intersections.append((x_intersect, threshold_value))
    
    right_intersections = []
    for i in range(peak_idx, len(smoothed_y) - 1):
        y_curr = float(smoothed_y[i])
        y_next = float(smoothed_y[i+1])
        
        if (y_curr >= threshold_value >= y_next) or (y_curr <= threshold_value <= y_next):
            denominator = y_next - y_curr
            
            if abs(denominator) > 1e-10:
                try:
                    t = (threshold_value - y_curr) / denominator
                    x_intersect = x_arr[i] + t * (x_arr[i+1] - x_arr[i])
                    right_intersections.append((x_intersect, threshold_value))
                except (OverflowError, FloatingPointError, ZeroDivisionError):
                    x_intersect = (x_arr[i] + x_arr[i+1]) / 2
                    right_intersections.append((x_intersect, threshold_value))
            else:
                x_intersect = (x_arr[i] + x_arr[i+1]) / 2
                right_intersections.append((x_intersect, threshold_value))
    
    intersections = []
    
    if left_intersections:
        leftmost = min(left_intersections, key=lambda point: point[0])
        intersections.append(leftmost)
    
    if right_intersections:
        rightmost = max(right_intersections, key=lambda point: point[0])
        intersections.append(rightmost)
    
    return intersections, peak_x, threshold_value

def sample_intensities_along_axis(image, com, principal_axis, step_size=1.0, max_steps=100, max_distance=None):
    y_max, x_max = image.shape
    
    intensities_forward = []
    intensities_backward = []
    positions_forward = []
    positions_backward = []
    distances_forward = []
    distances_backward = []
    
    directions = [1, -1]  
    
    for direction_idx, direction in enumerate(directions):
        current_point = com.copy()
        current_distance = 0
        
        for step in range(max_steps):
            y, x = int(round(current_point[0])), int(round(current_point[1]))
            
            if y < 0 or y >= y_max or x < 0 or x >= x_max:
                break
                
            intensity = image[y, x]
            
            if direction_idx == 0:  
                intensities_forward.append(intensity)
                positions_forward.append((y, x))
                distances_forward.append(current_distance)
            else:  
                intensities_backward.append(intensity)
                positions_backward.append((y, x))
                distances_backward.append(current_distance)
            
            next_point = current_point.copy()
            next_point[0] += direction * principal_axis[0] * step_size
            next_point[1] += direction * principal_axis[1] * step_size
            
            step_distance = np.sqrt(
                (next_point[0] - current_point[0])**2 + 
                (next_point[1] - current_point[1])**2
            )
            current_distance += step_distance
            
            if max_distance is not None and current_distance > max_distance:
                break
                
            current_point = next_point.copy()
    
    return {
        'intensities_forward': intensities_forward,
        'positions_forward': positions_forward,
        'distances_forward': distances_forward,
        'intensities_backward': intensities_backward,
        'positions_backward': positions_backward,
        'distances_backward': distances_backward
    }

def find_equal_area_threshold(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    if len(x_arr) < 3:
        return None, None, None
    
    peak_idx = np.argmax(y_arr)
    x_peak = x_arr[peak_idx]
    y_peak = y_arr[peak_idx]
    
    left_side = y_arr[:peak_idx]
    right_side = y_arr[peak_idx:]
        
    num_end_points = max(int(len(y_arr) * 0.1), 1)
    baseline = (np.mean(y_arr[:num_end_points]) + np.mean(y_arr[-num_end_points:])) / 2
    
    peak_base_threshold = baseline + 0.6 * (y_peak - baseline)
    
    left_base_idx = None
    for i in range(len(left_side)-1, -1, -1):
        if left_side[i] < peak_base_threshold:
            left_base_idx = i
    
    right_base_idx = None
    for i in range(len(right_side)):
        if right_side[i] < peak_base_threshold:
            right_base_idx = peak_idx + i
            
    if left_base_idx is None:
        left_base_idx = 0
    if right_base_idx is None:
        right_base_idx = len(y_arr) - 1
        
    y_base = max(peak_base_threshold, baseline)
    
    return x_peak, y_base, y_peak


def analyze_threshold_patterns_with_intensities(file_path, output_path='pattern_analysis.gif', 
                                       mask_color='red', alpha=0.5, start_slice=0, end_slice=None, 
                                       step=1, correlation_threshold=0.5):
    distances_um_list = []
    
    conversion = convert_to_microns(file_path)
    
    tiff_data = io.imread(file_path)
    
    if tiff_data.ndim < 3:
        raise ValueError("Input is not a stack, it's a single image")
    
    num_slices = tiff_data.shape[0]
    
    if end_slice is None:
        end_slice = num_slices
    else:
        end_slice = min(end_slice, num_slices)
    
    frames = []
    
    color_map = {
        'red': [1, 0, 0], 
        'green': [0, 1, 0], 
        'blue': [0, 0, 1], 
        'yellow': [1, 1, 0]
    }
        
    slice_indices = []
    correlation_scores = []
    pattern_results = []
    principal_axis_data = []
    endpoint_data = []
    intensity_profiles = []
    valid_slice_indices = []
    valid_distances = []
    left_points = []
    right_points = []
    
    for slice_idx in range(start_slice, end_slice, step):
        image = tiff_data[slice_idx]
        original_normalized = image / np.max(image) if np.max(image) > 0 else image
        
        image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
        denoised = ndimage.median_filter(image_eq, size=5)
        smoothed = ndimage.gaussian_filter(denoised, sigma=3)
        
        otsu_threshold = filters.threshold_otsu(smoothed)
        otsu_thresholded = smoothed > otsu_threshold
        
        selem = morphology.disk(3)
        cleaned = morphology.remove_small_objects(otsu_thresholded, min_size=100)
        closed = morphology.binary_closing(cleaned, selem)
        dilated = morphology.binary_dilation(closed, selem)
        opened = morphology.binary_opening(dilated, selem)
        initial_mask = opened
        
        labeled_mask = measure.label(initial_mask)
        regions = measure.regionprops(labeled_mask)
        
        center_y, center_x = np.array(image.shape) // 2
        center_mask = np.zeros_like(initial_mask, dtype=bool)
        
        if regions:
            min_dist = float('inf')
            center_region = None
            
            for region in regions:
                y, x = region.centroid
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    center_region = region
            
            if center_region:
                center_mask = labeled_mask == center_region.label
        
        secondary_threshold_mask = np.zeros_like(center_mask, dtype=bool)
        if np.any(center_mask):
            masked_image = image.copy()
            masked_image[~center_mask] = 0
            secondary_threshold_mask = (masked_image >= 390) & (masked_image <= 65535) & center_mask
            denoised = ndimage.median_filter(secondary_threshold_mask, size=5)
            secondary_threshold_mask = denoised
        
        correlation_analysis = analyze_thresholded_points_correlation(mask=secondary_threshold_mask)
        correlation_scores.append(correlation_analysis['correlation_score'])
        pattern_results.append(correlation_analysis)
        slice_indices.append(slice_idx)
        
        com, principal_axis, secondary_axis, eigenvalues = find_center_of_mass_and_principal_axis(secondary_threshold_mask)
        
        endpoints = []
        intensity_profile = None
        
        if correlation_analysis['has_pattern'] and com is not None and principal_axis is not None:
            endpoints = find_principal_axis_endpoints_in_mask(secondary_threshold_mask, com, 
                                                            principal_axis, step_size=1.0, max_steps=200)
            
            intensity_profile = sample_intensities_along_axis(
                image, com, principal_axis, step_size=1.0, max_steps=200, max_distance=None
            )
            
            print(f"\nSlice {slice_idx} Intensity profile along principal axis:")
        
        principal_axis_data.append({
            'center_of_mass': com,
            'principal_axis': principal_axis,
            'secondary_axis': secondary_axis,
            'eigenvalues': eigenvalues
        })
        
        endpoint_data.append(endpoints)
        intensity_profiles.append(intensity_profile)
        
        if intensity_profile is not None:
            all_distances = []
            all_intensities = []
            
            backward_distances = [-d for d in reversed(intensity_profile['distances_backward'])]
            backward_intensities = list(reversed(intensity_profile['intensities_backward']))
            
            all_distances = backward_distances + intensity_profile['distances_forward']
            all_intensities = backward_intensities + intensity_profile['intensities_forward']
            
            smoothed_intensities = savgol_filter(all_intensities, window_length=21, polyorder=2)
            
            peak_position, baseline, peak_height = find_equal_area_threshold(all_distances, smoothed_intensities)
            
            if peak_position is not None:
                intersections, _, _ = find_peak_intersections(all_distances, smoothed_intensities, baseline)
                
                if len(intersections) == 2 and com is not None and principal_axis is not None:
                    left_point_px = (
                        com[0] + intersections[0][0] * principal_axis[0],
                        com[1] + intersections[0][0] * principal_axis[1]
                    )
                    right_point_px = (
                        com[0] + intersections[1][0] * principal_axis[0],
                        com[1] + intersections[1][0] * principal_axis[1]
                    )

                    distance_px = np.sqrt((right_point_px[0] - left_point_px[0])**2 + 
                                        (right_point_px[1] - left_point_px[1])**2)
                    distance_um = distance_px * conversion

                    left_points.append(left_point_px)
                    right_points.append(right_point_px)
                    distances_um_list.append(distance_um)
                    valid_slice_indices.append(slice_idx)
                    valid_distances.append(distance_um)
    
    smoothed_distances = savgol_filter(valid_distances, window_length=min(21, len(valid_distances) if valid_distances else 1), polyorder=min(2, len(valid_distances)-1 if len(valid_distances) > 1 else 0)) if len(valid_distances) > 3 else valid_distances
    
    longest_slope_indices = find_longest_positive_slope(valid_slice_indices, smoothed_distances)
    
    longest_slope_slice_indices = set(valid_slice_indices[i] for i in longest_slope_indices) if longest_slope_indices else set()
    
    frames = []
    
    for slice_idx in range(start_slice, end_slice, step):
        image = tiff_data[slice_idx]
        original_normalized = image / np.max(image) if np.max(image) > 0 else image
        
        if original_normalized.ndim == 3:
            original_rgb = original_normalized.copy()
        else:
            original_rgb = np.stack([original_normalized, 
                                   original_normalized, 
                                   original_normalized], axis=2)
                
        fig = plt.figure(figsize=(8, 10))
        
        plt.imshow(original_rgb)
        plt.title(f"Slice {slice_idx} - Final Threshold")
        
        try:
            intensity_profile = intensity_profiles[slice_indices.index(slice_idx)]
            
            if intensity_profile is not None:                
                all_distances = []
                all_intensities = []
                
                backward_distances = [-d for d in reversed(intensity_profile['distances_backward'])]
                backward_intensities = list(reversed(intensity_profile['intensities_backward']))
                
                all_distances = backward_distances + intensity_profile['distances_forward']
                all_intensities = backward_intensities + intensity_profile['intensities_forward']
                
                smoothed_intensities = savgol_filter(all_intensities, window_length=21, polyorder=2)
                
                peak_position, baseline, peak_height = find_equal_area_threshold(all_distances, smoothed_intensities)
                
                if peak_position is not None:
                    intersections, _, _ = find_peak_intersections(all_distances, smoothed_intensities, baseline)
                                        
                    com = principal_axis_data[slice_indices.index(slice_idx)]['center_of_mass']
                    principal_axis = principal_axis_data[slice_indices.index(slice_idx)]['principal_axis']
                    
                    if len(intersections) == 2 and com is not None and principal_axis is not None:
                        left_point_px = (
                            com[0] + intersections[0][0] * principal_axis[0],
                            com[1] + intersections[0][0] * principal_axis[1]
                        )
                        right_point_px = (
                            com[0] + intersections[1][0] * principal_axis[0],
                            com[1] + intersections[1][0] * principal_axis[1]
                        )
                        
                        if slice_idx in longest_slope_slice_indices:
                            plt.plot(left_point_px[1], left_point_px[0], 'ro', markersize=10, alpha=0.8)
                            plt.plot(right_point_px[1], right_point_px[0], 'ro', markersize=10, alpha=0.8)

        except ValueError:
            pass

        plt.tight_layout()
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
    
    print("Processing gif...")
    #imageio.mimsave(output_path, frames, duration=0.5, loop=0)

    longest_slope_distances = [valid_distances[i] for i in longest_slope_slice_indices if i < len(valid_distances)]
    valid_left_points = [left_points[i] for i in longest_slope_slice_indices if i < len(left_points)]
    valid_right_points = [right_points[i] for i in longest_slope_slice_indices if i < len(right_points)]

    return {
        'slice_indices': slice_indices,
        'correlation_scores': correlation_scores,
        'pattern_results': pattern_results,
        'file_analyzed': file_path,
        'output_gif': output_path,
        'intensity_profiles': intensity_profiles,
        'valid_indices': list(longest_slope_slice_indices),
        'valid_distances': longest_slope_distances,
        'valid_left_points': valid_left_points,
        'valid_right_points': valid_right_points
    }

def find_longest_positive_slope(indices, distances, tolerance=0.02):
    indices = list(indices)
    distances = list(distances)
    
    if not indices or not distances:
        return []
        
    longest_slope = []
    current_slope = [0]
    
    max_distance = max(distances)
    min_distance = min(distances)
    normalized_tolerance = tolerance * (max_distance - min_distance)
    
    for i in range(len(indices) - 1):
        change = distances[i+1] - distances[i]
        
        if change > -normalized_tolerance:
            current_slope.append(i+1)
        else:
            if len(current_slope) > len(longest_slope):
                longest_slope = current_slope.copy()
            current_slope = [i+1]
    
    if len(current_slope) > len(longest_slope):
        longest_slope = current_slope.copy()
    
    return longest_slope

def analyze_thresholded_points_correlation(mask):
    points = np.argwhere(mask)
    
    if len(points) < 5: 
        return {
            'has_pattern': False,
            'correlation_score': 0,
            'clustering_score': 0,
            'linearity_score': 0,
            'compactness': 0
        }
    
    point_distances = distance.pdist(points)
    
    distance_variance = np.var(point_distances)
    max_possible_variance = np.var([0, np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)])
    normalized_variance = 1 - min(distance_variance / max_possible_variance, 1)
    
    dbscan = DBSCAN(eps=10, min_samples=3).fit(points)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = list(labels).count(-1) / len(labels) if labels.size > 0 else 1
    clustering_score = n_clusters * (1 - noise_ratio) * 2 / max(len(points) / 10, 1)
    clustering_score = min(clustering_score, 1)
    
    if len(points) > 2:
        try:
            r_value, _ = pearsonr(points[:, 0], points[:, 1])
            linearity_score = abs(r_value)
        except:
            linearity_score = 0
    else:
        linearity_score = 0
    
    if len(points) > 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume 
            compactness = len(points) / hull_area if hull_area > 0 else 0
            compactness = min(compactness / 0.1, 1)
        except:
            compactness = 0
    else:
        compactness = 0
    
    correlation_score = 0.4 * normalized_variance + 0.3 * clustering_score + 0.2 * linearity_score + 0.1 * compactness
    
    has_pattern = correlation_score > 0.5
    
    return {
        'has_pattern': has_pattern,
        'correlation_score': correlation_score,
        'clustering_score': clustering_score,
        'linearity_score': linearity_score,
        'compactness': compactness
    }

def find_center_of_mass_and_principal_axis(mask):
    if not np.any(mask):
        return None, None, None, None
    
    points = np.argwhere(mask)
    
    if len(points) < 3:
        center_of_mass = np.mean(points, axis=0)
        return center_of_mass, None, None, None
    
    center_of_mass = np.mean(points, axis=0)
    
    centered_points = points - center_of_mass
    
    cov_matrix = np.cov(centered_points, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    principal_axis = eigenvectors[:, 0]
    
    secondary_axis = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None
    
    return center_of_mass, principal_axis, secondary_axis, eigenvalues

def find_principal_axis_endpoints_in_mask(mask, com, principal_axis, step_size=1.0, max_steps=100):
    y_max, x_max = mask.shape
    
    endpoints = []
    directions = [1, -1]
    
    for direction in directions:
        current_point = com.copy()
        found_endpoint = False
        
        for step in range(1, max_steps):
            next_point = current_point.copy()
            next_point[0] += direction * principal_axis[0] * step_size
            next_point[1] += direction * principal_axis[1] * step_size
            
            y, x = int(round(next_point[0])), int(round(next_point[1]))
            
            if y < 0 or y >= y_max or x < 0 or x >= x_max:
                endpoints.append(current_point.copy())
                found_endpoint = True
                break
                
            if not mask[y, x]:
                endpoints.append(current_point.copy())
                found_endpoint = True
                break
                
            current_point = next_point.copy()
            
            if step == max_steps - 1:
                endpoints.append(current_point.copy())
                found_endpoint = True
        
        if not found_endpoint:
            endpoints.append(current_point.copy())
    
    return endpoints

def save_lengths_to_csv(all_file_results, output_path="lengths.csv"):
    with open(output_path, 'w') as f:
        for file_idx, results in enumerate(all_file_results):
            file_name = os.path.basename(results['file_analyzed']).replace(".tif", "")
            
            if file_idx > 0:
                f.write("\n\n")
                
            f.write(f"{file_name}\n")
            f.write("slice_index,length_um,left_point_y,left_point_x,right_point_y,right_point_x\n")
            
            valid_indices = results.get('valid_indices', [])
            valid_distances = results.get('valid_distances', [])
            valid_left_points = results.get('valid_left_points', [])
            valid_right_points = results.get('valid_right_points', [])
            
            print(f"File: {file_name}")
            print(f"Number of valid indices: {len(valid_indices)}")
            print(f"Number of valid distances: {len(valid_distances)}")
            print(f"Number of valid left points: {len(valid_left_points)}")
            print(f"Number of valid right points: {len(valid_right_points)}")
            
            for i, (idx, distance, left_point, right_point) in enumerate(zip(valid_indices, valid_distances, valid_left_points, valid_right_points)):
                adjusted_index = idx + 1
                f.write(f"{adjusted_index},{distance:.2f},{left_point[0]:.2f},{left_point[1]:.2f},{right_point[0]:.2f},{right_point[1]:.2f}\n")

if __name__ == "__main__":
    folder_path = "Movies"
    output_folder = "Processed_Results"
    all_results = []

    os.makedirs(output_folder, exist_ok=True)

    tif_files = glob.glob(os.path.join(folder_path, "*.tif"))

    for file_path in tif_files:
        print(f"Processing file: {file_path}")
        results = analyze_threshold_patterns_with_intensities(
            file_path,
            output_path=os.path.join(output_folder, f"{os.path.basename(file_path).replace('.tif', '')}_analysis.gif"),
            mask_color="red",
            start_slice=0,
            end_slice=None,
            correlation_threshold=0.5
        )
        
        print(f"Found {len(results.get('valid_slice_indices', []))} valid measurements in {file_path}")
        
        all_results.append(results)

    csv_output_path = os.path.join(output_folder, "lengths.csv")
    print(f"Saving length measurements to {csv_output_path}")
    save_lengths_to_csv(all_results, csv_output_path)
    print("CSV file saved successfully")