import pytesseract


def run_tesseract_ocr(roi_img):
    """
    Run Tesseract OCR on the given ROI image (which should already be preprocessed)
    and return:
      - The recognized text.
      - A bounding box (left, top, right, bottom) in roi_img coordinates (with an extra 5px padding)
        that covers all detected text.
    This version does not perform any additional preprocessing.
    """
    data = pytesseract.image_to_data(roi_img, output_type=pytesseract.Output.DICT)
    n = len(data['level'])
    boxes = []
    texts = []
    for i in range(n):
        text = data['text'][i].strip()
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1
        if text != "" and conf > 0:
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            boxes.append((left, top, left + width, top + height))
            texts.append(text)
    if not boxes:
        return "", None
    # Compute union of all bounding boxes
    all_left = min(b[0] for b in boxes)
    all_top = min(b[1] for b in boxes)
    all_right = max(b[2] for b in boxes)
    all_bottom = max(b[3] for b in boxes)
    pad = 5
    all_left = max(0, all_left - pad)
    all_top = max(0, all_top - pad)
    all_right = all_right + pad
    all_bottom = all_bottom + pad
    combined_text = " ".join(texts)
    return combined_text, (all_left, all_top, all_right, all_bottom)