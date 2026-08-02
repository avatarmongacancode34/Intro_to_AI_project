from pathlib import Path
import cv2



# 1. PROJECT PATH


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA = PROJECT_ROOT / "dataset" / "processed"



# 2. EXPECTED SETTINGS


EXPECTED_SIZE = (224, 224)

SUPPORTED_FORMATS = [".jpg"]



# 3. CHECK PROCESSED DATASET


total_images = 0
incorrect_size = []
unreadable_images = []
class_counts = {}


for class_folder in sorted(PROCESSED_DATA.iterdir()):

    if not class_folder.is_dir():
        continue

    count = 0

    for image_file in class_folder.iterdir():

        if image_file.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        total_images += 1
        count += 1

        image = cv2.imread(str(image_file))

        # Check if image can be read
        if image is None:
            unreadable_images.append(str(image_file))
            continue

        # Get dimensions
        height, width = image.shape[:2]

        if (width, height) != EXPECTED_SIZE:
            incorrect_size.append(
                f"{image_file.name}: {width}x{height}"
            )

    class_counts[class_folder.name] = count



# 4. DISPLAY RESULTS


print("\n==============================")
print("PREPROCESSING VALIDATION")
print("==============================")

print("\nTotal processed images:", total_images)

print("\nImages per class")
print("----------------")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")


print("\nIncorrect image sizes:", len(incorrect_size))
print("Unreadable images:", len(unreadable_images))



# 5. FINAL RESULT


if (
    total_images == 783
    and len(incorrect_size) == 0
    and len(unreadable_images) == 0
):

    print("\n PREPROCESSING VALIDATION PASSED")

else:

    print("\n PREPROCESSING VALIDATION FAILED")