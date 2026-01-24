from ultralytics import YOLO
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = 0  # Use GPU 0
else:
    print("No GPU detected, using CPU")
    device = 'cpu'

# Create a YOLOv12 model - starting with pretrained YOLOv12n
model = YOLO('yolo12n.pt')  # Already downloaded

# Train the model with your new dataset
results = model.train(
    data='Chimney Smoke Dataset.v6i.yolov12/data.yaml',  # Updated dataset path
    epochs=150,        # Increased epochs since you have more data
    imgsz=800,
    batch=16,          # Adjust based on GPU memory
    patience=20,       # Early stopping patience
    save=True,
    project='chimney_yolo12_training',  # ✅ Changed to avoid conflict
    name='exp1',
    verbose=True,
    device=device,
    workers=0,
    
    # Optimizations for larger dataset
    cos_lr=True,
    close_mosaic=10,
    
    # Data augmentation (can adjust based on your data)
    hsv_h=0.015,       # HSV hue augmentation
    hsv_s=0.7,         # HSV saturation
    hsv_v=0.4,         # HSV value
    degrees=10,        # Rotation
    translate=0.1,     # Translation
    scale=0.5,         # Scale
    flipud=0.0,        # Vertical flip (probably not for chimneys)
    fliplr=0.5,        # Horizontal flip
    mosaic=1.0,        # Mosaic augmentation
)

# Print results
print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Best model saved to: chimney_yolo12_training/exp1/weights/best.pt")
print(f"Results box mAP50: {results.box.map50:.4f}")
print(f"Results box mAP50-95: {results.box.map:.4f}")
print("="*60)

# Validate the best model
print("\nValidating best model...")
metrics = model.val()
print(f"Validation mAP50: {metrics.box.map50:.4f}")
print(f"Validation mAP50-95: {metrics.box.map:.4f}")