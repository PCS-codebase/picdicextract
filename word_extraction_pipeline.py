import os
import sys
import zipfile
import tempfile
import csv
from traceback import print_exc

import shutil

from roifile import ImagejRoi, ROI_TYPE
from PIL import Image, ImageDraw


import numpy as np
from shapely.geometry import Polygon
import rasterio.features
from affine import Affine

import cv2

from utils.filesystem import sanitize_filename
from utils.text import validate_text
from utils.image_processing import remove_background
from utils.tesseract import run_tesseract_ocr







def process_single_roi(roi, image):
    """
    Given an ImagejRoi object and a PIL Image, extract the ROI region.
    
    For rectangular ROIs:
      - Crop the image using the rectangle (plus 5px padding).
      
    For freehand ROIs:
      - Create a Shapely Polygon from the ROI coordinates.
      - Compute its bounding box (with 5px padding).
      - Rasterize the polygon to produce a binary mask.
      - Crop the image and mask to the ROI bounding box.
      - Apply the mask to isolate ROI pixels.
      
    Returns:
      (roi_type, bbox, roi_img)
      
      where bbox is in full image coordinates.
    """
    if roi.roitype == ROI_TYPE.RECT:
        left, top, right, bottom = roi.left, roi.top, roi.right, roi.bottom
        bbox = (max(0, left - 5), max(0, top - 5),
                min(image.width, right + 5), min(image.height, bottom + 5))
        roi_img = image.crop(bbox)
        return ("Rectangular", bbox, roi_img)
    elif roi.roitype == ROI_TYPE.FREEHAND:
        coords = roi.coordinates()
        if coords is None or len(coords) < 2:
            print("Insufficient coordinate data found for freehand ROI.")
            return ("Freehand", None, None)
        polygon = Polygon(coords)
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = (max(0, int(min_x) - 5),
                max(0, int(min_y) - 5),
                min(image.width, int(max_x) + 5),
                min(image.height, int(max_y) + 5))
        
        np_img = np.array(image)
        mask_full = rasterio.features.rasterize(
            [(polygon, 1)],
            out_shape=(image.height, image.width),
            transform=Affine.identity(),
            fill=0,
            dtype=np.uint8
        )
        cropped_mask = mask_full[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_image = np_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if cropped_image.ndim == 3:
            isolated = np.where(cropped_mask[..., None] == 1, cropped_image, 0)
        else:
            isolated = np.where(cropped_mask == 1, cropped_image, 0)
        roi_img = Image.fromarray(isolated.astype(np.uint8))
        return ("Freehand", bbox, roi_img)
    else:
        print(f"ROI type {roi.roitype} not specifically handled.")
        return (str(roi.roitype), None, None)

def process_zip_file(zip_path, images_folder, debug_folder, results):
    """
    Process a single .zip file containing ROI files for a given page:
      - Unzip ROI files.
      - Load the associated page image (derived by replacing 'roiset.zip' with '.png').
      - For each ROI file:
          - Parse and isolate the ROI region.
          - Create a modified version by removing background noise using adaptive thresholding.
          - Run Tesseract OCR on the modified image.
          - Validate the recognized text to ensure it contains only proper English words (with allowed exceptions).
          - If no valid text is detected:
              1. Try applying a 2px erosion on the Otsu-processed image.
              2. If still nothing, try OCR on the unmodified grayscale ROI.
          - Convert the OCR bounding box from ROI image coordinates to global image coordinates.
          - Create a side-by-side debug image showing the original ROI snippet (left) and the processed image (right, with OCR bounding box drawn).
          - Append structured results (including the OCR method used).
    """
    zip_basename = os.path.basename(zip_path)
    if zip_basename.lower().endswith("roiset.zip"):
        base_name = zip_basename[:-len("roiset.zip")]
    else:
        base_name = os.path.splitext(zip_basename)[0]
    image_filename = base_name + ".png"
    image_path = os.path.join(images_folder, image_filename)
    if not os.path.exists(image_path):
        print(f"Associated image {image_path} not found for {zip_basename}.")
        return

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return
        
        for file_name in os.listdir(tmpdirname):
            if file_name.lower().endswith(".roi"):
                roi_file_path = os.path.join(tmpdirname, file_name)
                try:
                    roi = ImagejRoi.fromfile(roi_file_path)
                    roi_type, bbox, roi_img = process_single_roi(roi, image)
                    if roi_img is None or bbox is None:
                        continue
                    
                    # Save a copy of the original ROI snippet
                    original_roi = roi_img.copy()
                    
                    # --- Step 1: Otsu thresholding ---
                    modified_roi = remove_background(roi_img)
                    ocr_text, ocr_box_rel = run_tesseract_ocr(modified_roi)
                    ocr_method = None
                    # Validate OCR text using the English dictionary and allowed exceptions
                    validated_text = validate_text(ocr_text, exceptions={"&", "-", "'"})
                    if validated_text.strip() and ocr_box_rel is not None:
                        ocr_text = validated_text
                        ocr_method = "otsu"
                    else:
                        # --- Step 2: Erosion on Otsu image (2px erosion) ---
                        otsu_array = np.array(modified_roi)
                        kernel = np.ones((2, 2), np.uint8)
                        eroded_array = cv2.erode(otsu_array, kernel, iterations=2)
                        eroded_img = Image.fromarray(eroded_array)
                        eroded_text, eroded_box = run_tesseract_ocr(eroded_img)
                        validated_text = validate_text(eroded_text, exceptions={"&", "-", "'"})
                        if validated_text.strip() and eroded_box is not None:
                            print("OCR succeeded with eroded Otsu image.")
                            ocr_text = validated_text
                            ocr_box_rel = eroded_box
                            ocr_method = "eroded"
                            modified_roi = eroded_img.convert("RGB")
                        else:
                            # --- Step 3: Fallback with unmodified grayscale image ---
                            fallback_img = roi_img.convert("L")  # Unmodified grayscale version
                            fallback_text, fallback_box = run_tesseract_ocr(fallback_img)
                            validated_text = validate_text(fallback_text, exceptions={"&", "-", "'"})
                            if validated_text.strip() and fallback_box is not None:
                                print("Fallback OCR succeeded with unmodified grayscale image.")
                                ocr_text = validated_text
                                ocr_box_rel = fallback_box
                                ocr_method = "fallback"
                                modified_roi = fallback_img.convert("RGB")
                            else:
                                ocr_method = "none"
                    
                    # Convert the OCR bounding box from ROI coordinates to global image coordinates
                    if ocr_box_rel is not None:
                        global_ocr_box = (bbox[0] + ocr_box_rel[0],
                                          bbox[1] + ocr_box_rel[1],
                                          bbox[0] + ocr_box_rel[2],
                                          bbox[1] + ocr_box_rel[3])
                    else:
                        global_ocr_box = None
                    
                    # Draw the OCR bounding box on the processed ROI image for visualization
                    if ocr_box_rel is not None:
                        draw = ImageDraw.Draw(modified_roi)
                        draw.rectangle(ocr_box_rel, outline="#FF0000", width=2)

                    # Create a side-by-side image: left = original ROI, right = processed ROI with bounding box
                    side_by_side_width = original_roi.width + modified_roi.width
                    side_by_side_height = max(original_roi.height, modified_roi.height)
                    side_by_side = Image.new("RGB", (side_by_side_width, side_by_side_height))
                    side_by_side.paste(original_roi, (0, 0))
                    side_by_side.paste(modified_roi, (original_roi.width, 0))

                    # Prepare the debug filename by including the sanitized OCR text (or "[none]" if empty)
                    word_for_filename = sanitize_filename(ocr_text) if ocr_text.strip() else "[none]"
                    debug_filename = f"{base_name}_{os.path.splitext(file_name)[0]}_{word_for_filename}.png"
                    debug_path = os.path.join(debug_folder, debug_filename)
                    side_by_side.save(debug_path)

                    results.append({
                        "page": base_name,
                        "roi_file": file_name,
                        "roi_type": roi_type,
                        "ocr_text": ocr_text,
                        "ocr_bbox": global_ocr_box,
                        "debug_image": debug_filename,
                        "ocr_method": ocr_method
                    })
                    
                    print(f"Processed {roi_file_path}: OCR text='{ocr_text}', OCR bbox={global_ocr_box}, method={ocr_method}")
                except Exception as e:
                    print(f"Error processing {roi_file_path}: {e}")
                    print_exc()

def main(annotated_folder, images_folder, debug_folder, output_csv):
    # Clear out the debug folder if it exists; otherwise, create it.
    if os.path.exists(debug_folder):
        for filename in os.listdir(debug_folder):
            file_path = os.path.join(debug_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(debug_folder)
    
    results = []
    for file_name in os.listdir(annotated_folder):
        if file_name.lower().endswith(".zip"):
            zip_path = os.path.join(annotated_folder, file_name)
            print(f"Processing zip file: {zip_path}")
            process_zip_file(zip_path, images_folder, debug_folder, results)
    
    # Write results to CSV.
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["page", "roi_file", "roi_type", "ocr_text", "ocr_bbox", "debug_image", "ocr_method"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)
    print(f"Results saved to {output_csv}")

    # Calculate and print OCR method percentages.
    total_rois = len(results)
    if total_rois > 0:
        otsu_count = sum(1 for r in results if r.get("ocr_method") == "otsu")
        eroded_count = sum(1 for r in results if r.get("ocr_method") == "eroded")
        fallback_count = sum(1 for r in results if r.get("ocr_method") == "fallback")
        none_count = sum(1 for r in results if r.get("ocr_method") == "none")
        print(f"Total ROIs processed: {total_rois}")
        print(f"Otsu step: {otsu_count} ({otsu_count/total_rois*100:.1f}%)")
        print(f"Eroded step: {eroded_count} ({eroded_count/total_rois*100:.1f}%)")
        print(f"Fallback step: {fallback_count} ({fallback_count/total_rois*100:.1f}%)")
        print(f"No word identified: {none_count} ({none_count/total_rois*100:.1f}%)")
    else:
        print("No ROIs were processed.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <annotated_folder> <images_folder> <debug_folder>")
        sys.exit(1)
    
    annotated_folder = sys.argv[1]  # e.g. './collinsdic_annotated'
    images_folder = sys.argv[2]       # e.g. './collinsdic_images'
    debug_folder = sys.argv[3]        # e.g. './debug'
    output_csv = "output.csv"
    
    main(annotated_folder, images_folder, debug_folder, output_csv)
