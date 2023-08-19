import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/Users/muhammadhamzasohail/miniforge3/bin/pytesseract'

# Load the image
image_path = '/Users/muhammadhamzasohail/Desktop/Scientific_notation_in_calculator_display_18.jpeg'

result = pytesseract.image_to_string(image_path, lang='', config='--psm 6')

print("Recognized Text:", result)
