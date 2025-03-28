import os
import sys
import shutil
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Converts each page of the given PDF file to an image saved in the output folder.
    Images are rendered at the specified dpi (default 300 dpi).
    """
    # Delete the folder if it already exists, then create a new one.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Calculate zoom factor for the desired dpi (default PDF resolution is 72 dpi)
    zoom_factor = dpi / 72.0
    matrix = fitz.Matrix(zoom_factor, zoom_factor)
    
    # Loop over each page and save as an image file
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        pix = page.get_pixmap(matrix=matrix)
        output_path = os.path.join(output_folder, f"page_{page_number+1:03d}.png")
        pix.save(output_path)
        print(f"Saved: {output_path}")
    
    print("All pages have been converted to images.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/file.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"Error: The file {pdf_path} does not exist.")
        sys.exit(1)
    
    # Create an output folder based on the PDF's base name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = f"{base_name}_images"
    
    convert_pdf_to_images(pdf_path, output_folder, dpi=300)

if __name__ == '__main__':
    main()
