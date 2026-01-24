from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained model
print("Loading trained YOLOv12 chimney detector...")
model = YOLO('chimney_yolo12_training/exp1/weights/best.pt')
print("✅ Model loaded successfully!\n")

# Path to your test image
test_image = "test_image5.png"  # Change this to your image path

print(f"Running inference on: {test_image}")
print("-" * 60)

# Run inference
results = model.predict(
    test_image,
    conf=0.25,        # Confidence threshold (adjust if needed)
    iou=0.45,         # IoU threshold for NMS
    imgsz=800,        # Same size as training
    verbose=True      # Show detailed output
)

# Process results
result = results[0]  # Get first (and only) result

print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

if len(result.boxes) > 0:
    print(f"✅ Found {len(result.boxes)} chimney(s)!\n")
    
    for i, box in enumerate(result.boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        
        print(f"Chimney {i+1}:")
        print(f"  Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Box size: {int(x2-x1)} x {int(y2-y1)} pixels\n")
else:
    print("❌ No chimneys detected")
    print("Try lowering the confidence threshold (currently 0.25)")

print("=" * 60)

# Visualize results
print("\nVisualizing results...")

# Get annotated image
annotated_img = result.plot()  # This returns BGR image with boxes drawn

# Convert BGR to RGB for matplotlib
annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(14, 10))
plt.imshow(annotated_img_rgb)
plt.axis('off')
plt.title(f'YOLOv12 Chimney Detection | {len(result.boxes)} chimney(s) detected', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Save annotated image
output_path = "chimney_detection_result.jpg"
cv2.imwrite(output_path, annotated_img)
print(f"\n💾 Saved annotated image to: {output_path}")

# Optional: Show detailed statistics
print("\n📊 Model Statistics:")
print(f"  Model: YOLOv12n")
print(f"  Parameters: 2.6M")
print(f"  Training mAP50: 90.5%")
print(f"  Inference speed: ~9.3ms per image")