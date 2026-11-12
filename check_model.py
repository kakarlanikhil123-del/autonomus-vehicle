from ultralytics import YOLO

print("Loading traffic-sign model...")

model = YOLO("best.pt")

print("Traffic-sign model loaded successfully!")
print("Classes:")
print(model.names)