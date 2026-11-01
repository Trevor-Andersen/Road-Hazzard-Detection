Python 3.12.7 (tags/v3.12.7:0b05ead, Oct  1 2024, 03:06:41) [MSC v.1941 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
... import csv
... import glob
... import shutil
... import numpy as np
... from PIL import Image
... from scipy.ndimage import convolve
... from dataclasses import dataclass
... from typing import List, Optional
... 
... # --- 1. Memory-Efficient Asset Class ---
... @dataclass
... class ImageAsset:
...     """Converts a raw image file path into a metadata asset with targeted on-demand analysis."""
...     file_path: str
...     filename: str
...     width: int
...     height: int
...     detected_imperfections: Optional[List[str]] = None
... 
...     def _get_road_roi(self) -> np.ndarray:
...         """
...         Slices the image on-demand to extract only the road surface.
...         Ignores horizon motion-blur and dashboard reflections.
...         """
...         with Image.open(self.file_path) as img:
...             gray = np.array(img.convert('L'))
...             
...         # Standard dashcam bounding box cropping rules for road focus
...         row_start = int(self.height * 0.55)  # Cut out sky/glare
...         row_end = int(self.height * 0.85)    # Cut out car hood
...         col_start = int(self.width * 0.20)   # Cut out oncoming traffic
...         col_end = int(self.width * 0.80)     # Cut out side scenery
...         
...         return gray[row_start:row_end, col_start:col_end]
... 
    @property
    def check_blurriness(self) -> float:
        """Calculates sharpness strictly within the road ROI using Laplacian variance."""
        road_pixels = self._get_road_roi()
        kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]])
        
        edges = convolve(road_pixels, kernel)
        return float(np.var(edges))


# --- 2. Automated Quality Gates ---
class PreLabelTests:
    @staticmethod
    def is_exposure_valid(asset: ImageAsset) -> bool:
        """Evaluates shadow/glare clipping limits strictly on the road surface."""
        road_pixels = asset._get_road_roi()
        total_pixels = road_pixels.size
        
        counts, _ = np.histogram(road_pixels, bins=256, range=(0, 256))
        too_dark = np.sum(counts[:20])     # Dark shadows/tunnel entry
        too_bright = np.sum(counts[240:])   # Bright sun reflection/headlights
        
        return (too_dark / total_pixels) < 0.50 and (too_bright / total_pixels) < 0.50

    @staticmethod
    def is_sharp_enough(asset: ImageAsset, min_blur_threshold: float = 12.0) -> bool:
        """Returns True if the road surface texture is sharp enough."""
        return asset.check_blurriness >= min_blur_threshold


# --- 3. Pipeline Orchestrator ---
def process_and_sort_fileset(directory_path: str, limit: int = 120) -> List[ImageAsset]:
    """Discovers images, logs failures with specific reasons, and exports a CSV index."""
    target_dir = os.path.normpath(directory_path)
    rejected_dir = os.path.join(target_dir, "rejected_frames")
    csv_output_path = os.path.join(target_dir, "valid_assets_summary.csv")
    
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(target_dir, ext)))
        if len(image_paths) >= limit:
            break

    selected_paths = image_paths[:limit]
    print(f"Windows Path Verified: {target_dir}")
    print(f"Processing a batch limit of {len(selected_paths)} images...")

    passed_assets = []
    rejected_count = 0

    for path in selected_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            
            asset = ImageAsset(
                file_path=path,
                filename=os.path.basename(path),
                width=w,
                height=h
            )
            
            # Run test checks individually to diagnose the failure type
            exposure_ok = PreLabelTests.is_exposure_valid(asset)
            sharp_ok = PreLabelTests.is_sharp_enough(asset, min_blur_threshold=12.0)
            
            if exposure_ok and sharp_ok:
                passed_assets.append(asset)
            else:
                # Lazy create the rejected directory only if a frame actually fails
                if not os.path.exists(rejected_dir):
                    os.makedirs(rejected_dir)
                
                # Determine failure tag
                if not exposure_ok and not sharp_ok:
                    reason = "FAILED_BOTH"
                elif not exposure_ok:
                    reason = "FAILED_EXPOSURE"
                else:
                    reason = "FAILED_BLUR"
                
                # Modify output filename to include failure reason
                name_parts = os.path.splitext(asset.filename)
                new_filename = f"{name_parts[0]}_{reason}{name_parts[1]}"
                
                # Copy to subfolder with preserving metadata
                shutil.copy2(path, os.path.join(rejected_dir, new_filename))
                rejected_count += 1
                
        except Exception as e:
            print(f"Error reading image {os.path.basename(path)}: {e}")
            rejected_count += 1

    # --- 4. CSV Text Summary Generation ---
    if passed_assets:
        try:
            with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['filename', 'width', 'height', 'file_path']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writeheader()
                for valid_asset in passed_assets:
                    writer.writerow({
                        'filename': valid_asset.filename,
                        'width': valid_asset.width,
                        'height': valid_asset.height,
                        'file_path': valid_asset.file_path
                    })
            print(f" -> CSV Manifest saved to: '{csv_output_path}'")
        except IOError as e:
            print(f"Warning: Could not write CSV summary file: {e}")
    else:
        print(" -> No CSV summary written because 0 assets passed the quality gates.")

    print(f"\nPipeline Summary:")
    print(f" -> Clean Assets (Returned): {len(passed_assets)}")
    print(f" -> Rejected (Copied to '\\rejected_frames'): {rejected_count}")
    
    return passed_assets


# --- Windows Execution Example ---
if __name__ == "__main__":
    # Raw string mapping to your windows dataset folder
    DATASET_DIR = r"C:\Users\ander\Road Defects.v1i.yolov11\Images" 
    
    if os.path.exists(DATASET_DIR):
        valid_assets = process_and_sort_fileset(DATASET_DIR, limit=120)
    else:
        fallback_path = os.path.join(".", "Road Defects.v1i.yolov11\Images")
        if not os.path.exists(fallback_path):
            os.makedirs(fallback_path)
            print(f"Created a dummy folder at '{os.path.abspath(fallback_path)}'. Please add your images there.")
        else:
            valid_assets = process_and_sort_fileset(fallback_path, limit=120)
