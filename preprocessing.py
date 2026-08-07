from pathlib import Path
import cv2
import numpy as np
import shutil

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "dataset" / "processed"


# ==========================================
# 2. SETTINGS
# ==========================================

IMAGE_SIZE = 224

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]


# ==========================================
# 3. RESIZE IMAGE WHILE PRESERVING
#    ASPECT RATIO
# ==========================================

def resize_with_padding(image, size=224):

    height, width = image.shape[:2]

    # Calculate scale
    scale = min(size / width, size / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create white square canvas
    canvas = np.ones(
        (size, size, 3),
        dtype=np.uint8
    ) * 255

    # Center the image
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


# ==========================================
# 4. PROCESS DATASET
# ==========================================

def process_dataset():

    # Check raw dataset
    if not RAW_DATA.exists():
        print(f"ERROR: Dataset not found: {RAW_DATA}")
        return

    # Remove old processed dataset
    if PROCESSED_DATA.exists():

        print("Removing old processed dataset...")

        shutil.rmtree(PROCESSED_DATA)

    # Create new processed folder
    PROCESSED_DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_count = 0
    failed_count = 0
    class_count = 0

    # ------------------------------------------
    # Go through every class
    # ------------------------------------------

    class_folders = sorted([
        folder
        for folder in RAW_DATA.iterdir()
        if folder.is_dir()
    ])

    print("\n==========================================")
    print("STARTING DATASET PREPROCESSING")
    print("==========================================")

    print(f"Classes found: {len(class_folders)}")
    print(f"Input size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    # ------------------------------------------
    # Process each class
    # ------------------------------------------

    for class_folder in class_folders:

        class_count += 1

        output_folder = (
            PROCESSED_DATA / class_folder.name
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        class_processed = 0

        # --------------------------------------
        # Process every image
        # --------------------------------------

        for image_file in sorted(class_folder.iterdir()):

            # Ignore unsupported files
            if image_file.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            # Read image
            image = cv2.imread(str(image_file))

            # Check if image was readable
            if image is None:

                print(
                    f"Could not read: {image_file}"
                )

                failed_count += 1
                continue

            # Resize and pad
            processed_image = resize_with_padding(
                image,
                IMAGE_SIZE
            )

            # ----------------------------------
            # IMPORTANT:
            # Prevent filename collisions
            # ----------------------------------

            extension = image_file.suffix.lower().replace(".", "")

            output_filename = (
                f"{image_file.stem}_{extension}.jpg"
            )

            output_file = (
                output_folder / output_filename
            )

            # Save processed image
            success = cv2.imwrite(
                str(output_file),
                processed_image
            )

            if success:

                processed_count += 1
                class_processed += 1

            else:

                print(
                    f"Could not save: {image_file}"
                )

                failed_count += 1

        print(
            f"[{class_count}/{len(class_folders)}] "
            f"{class_folder.name}: "
            f"{class_processed} images"
        )

    # ==========================================
    # 5. FINAL SUMMARY
    # ==========================================

    print("\n==========================================")
    print("PREPROCESSING COMPLETE")
    print("==========================================")

    print(
        f"Classes processed     : {class_count}"
    )

    print(
        f"Successfully processed: {processed_count}"
    )

    print(
        f"Failed                : {failed_count}"
    )

    print(
        f"Output directory      : {PROCESSED_DATA}"
    )

    print("==========================================")


# ==========================================
# 6. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    process_dataset()