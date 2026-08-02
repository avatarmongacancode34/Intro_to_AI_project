from pathlib import Path
import cv2
import numpy as np





# 1. PROJECT PATHS


# Find the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent

# Original dataset
RAW_DATA = PROJECT_ROOT / "dataset" / "raw"

# Where processed images will be saved
PROCESSED_DATA = PROJECT_ROOT / "dataset" / "processed"



# 2. SETTINGS


# CNN input size
IMAGE_SIZE = 224

# Supported image formats
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]



# 3. RESIZE IMAGE WHILE PRESERVING ASPECT RATIO


def resize_with_padding(image, size=224):

    # Get original height and width
    height, width = image.shape[:2]

    # Calculate scale needed to fit image inside
    # the target square
    scale = min(size / width, size / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize while keeping the original aspect ratio
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create a square canvas
    canvas = np.ones(
    (size, size, 3),
    dtype=np.uint8
    ) * 255

    # Calculate position for centered image
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    # Place resized image in the center
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas



# 4. PROCESS THE DATASET


def process_dataset():

    # Create processed folder if it doesn't exist
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    failed_count = 0

    # Go through every symbol folder
    for class_folder in RAW_DATA.iterdir():

        if not class_folder.is_dir():
            continue

        # Create matching folder in processed/
        output_folder = PROCESSED_DATA / class_folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Process every image
        for image_file in class_folder.iterdir():

            # Ignore unsupported files
            if image_file.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            # Read image
            image = cv2.imread(str(image_file))

            # Check if OpenCV successfully read it
            if image is None:
                print(f"Could not read: {image_file}")
                failed_count += 1
                continue

            # Resize while maintaining aspect ratio
            processed_image = resize_with_padding(
                image,
                IMAGE_SIZE
            )

            # Save as JPG
            # Create a unique output filename
            output_filename = (
                f"{image_file.stem}_{image_file.suffix[1:]}.jpg"
                )

            output_file = output_folder / output_filename

            success = cv2.imwrite(
                str(output_file),
                processed_image
            )

            if success:
                processed_count += 1
                print(f"Processed: {image_file.name}")
            else:
                failed_count += 1
                print(f"Could not save: {image_file.name}")

    
    # 5. SUMMARY
    

    print("\n==============================")
    print("PREPROCESSING COMPLETE")
    print("==============================")

    print("Successfully processed:", processed_count)
    print("Failed:", failed_count)



# 6. RUN THE PROGRAM


if __name__ == "__main__":
    process_dataset()