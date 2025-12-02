import os

DATASET_PATH = r"C:\Users\VAISHNAVI\Downloads\archive (6)\Dataset"

found_images = []
found_csvs = []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            found_images.append(os.path.join(root, file))
        elif file.lower().endswith('.csv'):
            found_csvs.append(os.path.join(root, file))

print("🖼 Found images:", len(found_images))
print("📊 Found CSVs:", len(found_csvs))

if len(found_images):
    print("\nSample image file:", found_images[0])
if len(found_csvs):
    print("\nSample CSV file:", found_csvs[0])
