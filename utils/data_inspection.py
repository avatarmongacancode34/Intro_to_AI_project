from pathlib import Path
import cv2



# 1. Locate the project and raw dataset


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "dataset" / "raw"



# 2. Get all symbol folders


symbol_folders = [
    folder for folder in RAW_DATA.iterdir()
    if folder.is_dir()
]



# 3. Display the classes


print("Adinkra Symbol Classes")
print("----------------------")

for folder in sorted(symbol_folders):
    print(folder.name)



# 4. Count images in each class


print("\nImages per Class")
print("----------------")

total_images = 0

for folder in sorted(symbol_folders):

    images = [
        file for file in folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    number_of_images = len(images)

    print(f"{folder.name}: {number_of_images}")

    total_images += number_of_images



# 5. Display total number of images

print("\nTotal Images")
print("------------")
print(total_images)


# 6. Inspect image formats and dimensions


print("\nImage Inspection")
print("----------------")

formats = {}
dimensions = {}
corrupted_images = []

for folder in sorted(symbol_folders):

    for file in folder.iterdir():

        if file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        # Count image formats
        extension = file.suffix.lower()
        formats[extension] = formats.get(extension, 0) + 1

        # Try to read the image
        image = cv2.imread(str(file))

        if image is None:
            corrupted_images.append(str(file))
            continue

        # Record image dimensions
        height, width = image.shape[:2]

        dimension = (width, height)

        dimensions[dimension] = dimensions.get(dimension, 0) + 1



# 7. Display formats


print("\nImage Formats")
print("-------------")

for extension, count in formats.items():
    print(f"{extension}: {count}")



# 8. Display dimensions


print("\nImage Dimensions")
print("----------------")

for dimension, count in sorted(dimensions.items()):
    print(f"{dimension}: {count}")



# 9. Display corrupted images


print("\nCorrupted / Unreadable Images")
print("-----------------------------")

print("Number of unreadable images:", len(corrupted_images))

for image in corrupted_images:
    print(image)