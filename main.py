import pytesseract
from PIL import Image

# Provide the path to your Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'gauage.png'
image = Image.open(image_path)

# Perform OCR on the image
result = pytesseract.image_to_string(image, config='--psm 6')

# Print the recognized text
print("Recognized Numbers:", result)
