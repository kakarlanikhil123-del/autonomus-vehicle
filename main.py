YOLO_MODEL = "best.pt"

from ultralytics import YOLO

model = YOLO("best.pt")

YOLO("best.pt")


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH