from ultralytics import YOLO

print("Loading Indian traffic-sign model...")

model = YOLO("indian_traffic.pt")

print("Model loaded successfully!")
print("Number of classes:", len(model.names))
print("Classes:")
print(model.names)
