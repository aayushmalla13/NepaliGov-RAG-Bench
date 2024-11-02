import cv2
import pytesseract
from PIL import Image

class OCRPipeline:
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6 -l nep+eng'
    
    def extract_text(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, config=self.tesseract_config)
        return text
    
    def preprocess_image(self, image_path):
        # Image preprocessing for better OCR
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
