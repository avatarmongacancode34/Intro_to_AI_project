import os
from pathlib import Path
from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset/raw"

LOW_COUNT_WARNING = 20
VERY_LOW_COUNT_WARNING = 10

# ==========================================
# STORAGE
# ==========================================

class_counts = {}
image_formats = {}
image_dimensions = {}

unreadable_images = []
small_images = []

# Used to detect filename collisions
filename_stems = {}

total_images = 0

# ==========================================
# INSPECT DATASET
# ==========================================

dataset_path = Path(DATASET_PATH)

if not dataset_path.exists():
    print(f"ERROR: Dataset folder not found: {DATASET_PATH}")
    exit()

classes = sorted([
    folder.name
    for folder in dataset_path.iterdir()
    if folder.is_dir()
])

print("\nAdinkra Dataset Inspection")
print("==========================")

print(f"Dataset path: {DATASET_PATH}")
print(f"Number of classes: {len(classes)}")

# ==========================================
# LOOP THROUGH CLASSES
# ==========================================

for class_name in classes:

    class_path = dataset_path / class_name

    class_count = 0

    for file_path in class_path.iterdir():

        if not file_path.is_file():
            continue

        # Only process image files
        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1
        class_count += 1

        # ----------------------------------
        # FORMAT
        # ----------------------------------

        extension = file_path.suffix.lower()

        image_formats[extension] = image_formats.get(extension, 0) + 1

        # ----------------------------------
        # FILENAME STEM
        # ----------------------------------

        stem = file_path.stem.lower()

        if stem not in filename_stems:
            filename_stems[stem] = []

        filename_stems[stem].append(str(file_path))

        # ----------------------------------
        # OPEN IMAGE
        # ----------------------------------

        try:

            with Image.open(file_path) as image:

                width, height = image.size

                dimension = (width, height)

                image_dimensions[dimension] = (
                    image_dimensions.get(dimension, 0) + 1
                )

                # ----------------------------------
                # VERY SMALL IMAGE CHECK
                # ----------------------------------

                if width < 100 or height < 100:

                    small_images.append({
                        "file": str(file_path),
                        "class": class_name,
                        "size": dimension
                    })

        except Exception as e:

            unreadable_images.append({
                "file": str(file_path),
                "error": str(e)
            })

    class_counts[class_name] = class_count


# ==========================================
# CLASS INFORMATION
# ==========================================

print("\nImages per Class")
print("----------------")

for class_name, count in class_counts.items():

    warning = ""

    if count < VERY_LOW_COUNT_WARNING:
        warning = "  ⚠ VERY LOW"

    elif count < LOW_COUNT_WARNING:
        warning = "  ⚠ LOW"

    print(f"{class_name}: {count}{warning}")


# ==========================================
# TOTAL IMAGES
# ==========================================

print("\nTotal Images")
print("------------")
print(total_images)


# ==========================================
# IMAGE FORMATS
# ==========================================

print("\nImage Formats")
print("-------------")

for extension, count in sorted(image_formats.items()):

    print(f"{extension}: {count}")


# ==========================================
# IMAGE DIMENSIONS
# ==========================================

print("\nImage Dimensions")
print("----------------")

print(f"Unique dimensions found: {len(image_dimensions)}")

# Show the most common dimensions

sorted_dimensions = sorted(
    image_dimensions.items(),
    key=lambda x: x[1],
    reverse=True
)

for dimension, count in sorted_dimensions[:20]:

    print(f"{dimension}: {count}")


# ==========================================
# SMALL IMAGE INSPECTION
# ==========================================

print("\nSmall Image Inspection")
print("----------------------")

very_small = []      # Below 50x50
small = []           # 50x50 to below 100x100
acceptable = []      # 100x100 and above

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)

    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            with Image.open(filepath) as img:
                width, height = img.size

                smallest_dimension = min(width, height)

                if smallest_dimension < 50:
                    very_small.append(
                        (class_name, filename, width, height, filepath)
                    )

                elif smallest_dimension < 100:
                    small.append(
                        (class_name, filename, width, height, filepath)
                    )

                else:
                    acceptable.append(
                        (class_name, filename, width, height, filepath)
                    )

        except Exception:
            pass


print("Very small images (<50 pixels):", len(very_small))
print("Small images (50–99 pixels):   ", len(small))
print("Acceptable images (100+ pixels):", len(acceptable))


# ==========================================
# SHOW VERY SMALL IMAGES
# ==========================================

print("\nVERY SMALL IMAGES (<50 pixels)")
print("-------------------------------")

for class_name, filename, width, height, filepath in very_small:
    print(
        f"{class_name} | "
        f"{filename} | "
        f"{width}x{height} | "
        f"{filepath}"
    )


# ==========================================
# SHOW SMALL IMAGES
# ==========================================

print("\nSMALL IMAGES (50–99 pixels)")
print("---------------------------")

for class_name, filename, width, height, filepath in small:
    print(
        f"{class_name} | "
        f"{filename} | "
        f"{width}x{height} | "
        f"{filepath}"
    )
# ==========================================
# UNREADABLE IMAGES
# ==========================================

print("\nCorrupted / Unreadable Images")
print("-----------------------------")

print(
    f"Number of unreadable images: "
    f"{len(unreadable_images)}"
)

if unreadable_images:

    for image in unreadable_images[:20]:

        print(image["file"])
        print("Error:", image["error"])


# ==========================================
# DUPLICATE FILENAME STEMS
# ==========================================

print("\nFilename Collisions")
print("-------------------")

collisions = {
    stem: files
    for stem, files in filename_stems.items()
    if len(files) > 1
}

print(
    f"Filename stems appearing more than once: "
    f"{len(collisions)}"
)

if collisions:

    print("\nExamples:")

    for stem, files in list(collisions.items())[:20]:

        print(f"\n{stem}")

        for file in files:

            print(f"  {file}")


# ==========================================
# CLASS IMBALANCE
# ==========================================

print("\nClass Imbalance Summary")
print("-----------------------")

counts = list(class_counts.values())

if counts:

    smallest = min(counts)
    largest = max(counts)

    smallest_classes = [
        name
        for name, count in class_counts.items()
        if count == smallest
    ]

    largest_classes = [
        name
        for name, count in class_counts.items()
        if count == largest
    ]

    print(f"Smallest class: {smallest} images")
    print(
        "Class(es):",
        ", ".join(smallest_classes)
    )

    print(f"Largest class: {largest} images")
    print(
        "Class(es):",
        ", ".join(largest_classes)
    )

    print(
        f"Imbalance ratio: "
        f"{largest / smallest:.2f}:1"
    )


# ==========================================
# CLASSES NEEDING ATTENTION
# ==========================================

print("\nClasses Needing Attention")
print("-------------------------")

very_low_classes = [
    (name, count)
    for name, count in class_counts.items()
    if count < VERY_LOW_COUNT_WARNING
]

low_classes = [
    (name, count)
    for name, count in class_counts.items()
    if VERY_LOW_COUNT_WARNING <= count < LOW_COUNT_WARNING
]

if very_low_classes:

    print("\nVERY LOW (<10 images):")

    for name, count in very_low_classes:

        print(f"  {name}: {count}")

if low_classes:

    print("\nLOW (10-19 images):")

    for name, count in low_classes:

        print(f"  {name}: {count}")

if not very_low_classes and not low_classes:

    print("No classes below 20 images.")


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n================================")
print("FINAL DATASET SUMMARY")
print("================================")

print(f"Classes              : {len(classes)}")
print(f"Total images         : {total_images}")
print(f"Unreadable images    : {len(unreadable_images)}")
print(f"Very small images    : {len(small_images)}")
print(f"Filename collisions  : {len(collisions)}")

print("\nInspection complete.")