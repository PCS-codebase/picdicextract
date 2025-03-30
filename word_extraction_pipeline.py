import os
import sys
import zipfile
import tempfile
import csv
from traceback import print_exc
import shutil
from roifile import ImagejRoi, ROI_TYPE
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from shapely.geometry import Polygon
import rasterio.features
from affine import Affine
import cv2
from collections import Counter  # added import
from utils.filesystem import sanitize_filename
from utils.text import validate_text
from utils.image_processing import remove_background
from utils.tesseract import run_tesseract_ocr

def process_single_roi(roi, image):
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
    zip_basename = os.path.basename(zip_path)
    base_name = zip_basename[:-len("roiset.zip")] if zip_basename.lower().endswith("roiset.zip") else os.path.splitext(zip_basename)[0]
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

                    original_roi = roi_img.copy()

                    def ocr_strategies(original_img):
                        otsu_img = remove_background(original_img)
                        yield "otsu", otsu_img

                        otsu_array = np.array(otsu_img)
                        kernel = np.ones((2, 2), np.uint8)
                        eroded_array = cv2.erode(otsu_array, kernel, iterations=2)
                        eroded_img = Image.fromarray(eroded_array)
                        yield "eroded", eroded_img.convert("RGB")

                        fallback_img = original_img.convert("L")
                        yield "fallback", fallback_img.convert("RGB")

                        cropped_img = original_img.convert("L")
                        width, height = cropped_img.size
                        crop_top = int(height * 0.10)
                        cropped_region = cropped_img.crop((0, crop_top, width, height))
                        yield "croptop10", cropped_region.convert("RGB")
                        
                        cropped_img = original_img.convert("L")
                        width, height = cropped_img.size
                        crop_top = int(height * 0.20)
                        cropped_region = cropped_img.crop((0, crop_top, width, height))
                        yield "croptop20", cropped_region.convert("RGB")


                    ocr_text = ""
                    ocr_box_rel = None
                    ocr_method = "none"
                    attempt_images = []

                    font = ImageFont.load_default()

                    for method, processed_img in ocr_strategies(roi_img):
                        text, box = run_tesseract_ocr(processed_img)
                        text_validity = validate_text(text, exceptions={"&", "-", "'"})
                        debug_img = processed_img.copy()
                        draw = ImageDraw.Draw(debug_img)
                        if box:
                            draw.rectangle(box, outline="#FF0000", width=2)
                        draw.text((2, 2), method, fill="yellow", font=font)
                        attempt_images.append(debug_img)

                        if text_validity and box is not None:
                            ocr_text = text
                            ocr_box_rel = box
                            ocr_method = method
                            print(f"OCR succeeded with {method} strategy.")
                            break

                    global_ocr_box = (bbox[0] + ocr_box_rel[0],
                                      bbox[1] + ocr_box_rel[1],
                                      bbox[0] + ocr_box_rel[2],
                                      bbox[1] + ocr_box_rel[3]) if ocr_box_rel else None

                    combined_images = [original_roi] + attempt_images
                    total_width = sum(img.width for img in combined_images)
                    max_height = max(img.height for img in combined_images)
                    composite = Image.new("RGB", (total_width, max_height))
                    x_offset = 0
                    for img in combined_images:
                        composite.paste(img, (x_offset, 0))
                        x_offset += img.width

                    word_for_filename = sanitize_filename(ocr_text) if ocr_text.strip() else "[none]"
                    debug_filename = f"{base_name}_{os.path.splitext(file_name)[0]}_{word_for_filename}.png"
                    
                    # Create a subfolder based on the final OCR strategy.
                    subfolder = os.path.join(debug_folder, ocr_method)
                    if not os.path.exists(subfolder):
                        os.makedirs(subfolder)
                    debug_path = os.path.join(subfolder, debug_filename)
                    composite.save(debug_path)

                    results.append({
                        "page": base_name,
                        "roi_file": file_name,
                        "roi_type": roi_type,
                        "ocr_text": ocr_text,
                        "ocr_bbox": global_ocr_box,
                        "debug_image": os.path.join(ocr_method, debug_filename),
                        "ocr_method": ocr_method
                    })

                    print(f"Processed {roi_file_path}: OCR text='{ocr_text}', OCR bbox={global_ocr_box}, method={ocr_method}")
                except Exception as e:
                    print(f"Error processing {roi_file_path}: {e}")
                    print_exc()

def main(annotated_folder, images_folder, debug_folder, output_csv):
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

    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["page", "roi_file", "roi_type", "ocr_text", "ocr_bbox", "debug_image", "ocr_method"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)
    print(f"Results saved to {output_csv}")

    total_rois = len(results)
    if total_rois > 0:
        # Use Counter to automatically count occurrences of each OCR method.
        counter = Counter(r.get("ocr_method", "none") for r in results)
        for method, count in counter.items():
            print(f"{method.capitalize()} step: {count} ({count/total_rois*100:.1f}%)")
    else:
        print("No ROIs were processed.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <annotated_folder> <images_folder> <debug_folder>")
        sys.exit(1)

    annotated_folder = sys.argv[1]
    images_folder = sys.argv[2]
    debug_folder = sys.argv[3]
    output_csv = "output.csv"

    main(annotated_folder, images_folder, debug_folder, output_csv)
