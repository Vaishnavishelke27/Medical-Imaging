import os

DATASET_PATH = r"C:\Users\VAISHNAVI\Downloads\archive (6)\Dataset\Brain Tumor CT scan Images"

if os.path.exists(DATASET_PATH):
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    print("Classes:", classes)
    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{cls}: {count} images")
else:
    print("Dataset path not found.")
