from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "dataset" / "raw"


for symbol_folder in RAW_DATA.iterdir():

    if not symbol_folder.is_dir():
        continue

    for image_file in symbol_folder.iterdir():

        if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_file))

        if image is None:
            continue

        height, width = image.shape[:2]

        # Look specifically for extremely tiny images
        if width <= 32 and height <= 32:

            print(f"\nClass: {symbol_folder.name}")
            print(f"File: {image_file.name}")
            print(f"Size: {width} x {height}")

            # Enlarge the tiny image so we can see it
            enlarged = cv2.resize(
                image,
                (300, 300),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(
                f"{symbol_folder.name} - {image_file.name}",
                enlarged
            )

            cv2.waitKey(0)
            cv2.destroyAllWindows()