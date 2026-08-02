from pathlib import Path
import cv2


# --------------------------------------------------
# 1. Locate the raw dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "dataset" / "raw"


# --------------------------------------------------
# 2. Find images smaller than 100 x 100 pixels
# --------------------------------------------------

small_images = []

for symbol_folder in RAW_DATA.iterdir():

    if not symbol_folder.is_dir():
        continue

    for image_file in symbol_folder.iterdir():

        if image_file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        image = cv2.imread(str(image_file))

        # Skip images OpenCV cannot read
        if image is None:
            continue

        height, width = image.shape[:2]

        if width < 100 or height < 100:

            small_images.append({
                "class": symbol_folder.name,
                "file": image_file.name,
                "width": width,
                "height": height,
                "path": str(image_file)
            })


# --------------------------------------------------
# 3. Display the results
# --------------------------------------------------

print("Small Images (< 100 x 100)")
print("--------------------------")

print("Number of small images:", len(small_images))

for image in small_images:

    print(
        f"{image['class']} | "
        f"{image['file']} | "
        f"{image['width']} x {image['height']}"
    )

