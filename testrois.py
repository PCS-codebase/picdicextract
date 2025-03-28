import os
import sys
import roifile
from traceback import print_exc
from roifile import ImagejRoi, ROI_TYPE


def process_roi_file(file_path):
    # Read the ROI file
    roi = ImagejRoi.fromfile(file_path)
    roi_type = roi.roitype
    
    print(f"File: {os.path.basename(file_path)}")
    
    # Check if ROI is a rectangle (rectangular)
    if roi_type == ROI_TYPE.RECT:
        left = roi.left
        top = roi.top
        right = roi.right
        bottom = roi.bottom
        print("Type: Rectangular")
        print(f"Coordinates: Top-Left ({left}, {top}), Bottom-Right ({right}, {bottom})")
    
    # For freehand selections, ImageJ typically saves them as polygons/freehand
    elif roi_type == ROI_TYPE.FREEHAND:
        print("Type: Freehand")
        coords = roi.coordinates()
        if coords is not None:
            # print("Coordinates:")
            # for x, y in coords:
            #     print(f"({x}, {y})")
            print(f"Found {len(coords)} points.")
            if len(coords) >= 2:
                start_coord = coords[0]
                end_coord =  coords [-1]
                print(f"first point: {start_coord[0]}, {start_coord[1]}")
                print(f"last point: {end_coord[0]}, {end_coord[1]}")
        else:
            print("No coordinate data found for freehand ROI.")
    
    else:
        print(f"Type: {roi_type} (not specifically handled)")
        print(roi)
    
    print("-" * 40)

def main(folder_path):
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".roi"):
            file_path = os.path.join(folder_path, file_name)
            try:
                process_roi_file(file_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <roi_folder_path>")
        sys.exit(1)
    
    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)
    
    main(folder)
