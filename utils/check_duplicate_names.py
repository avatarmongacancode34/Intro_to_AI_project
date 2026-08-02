from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA = PROJECT_ROOT / "dataset" / "raw"


for class_folder in sorted(RAW_DATA.iterdir()):

    if not class_folder.is_dir():
        continue

    names = {}
    duplicates = []

    for image_file in class_folder.iterdir():

        if image_file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png"
        ]:
            continue

        # Get filename without extension
        stem = image_file.stem.lower()

        if stem in names:
            duplicates.append(
                f"{names[stem].name} <-> {image_file.name}"
            )
        else:
            names[stem] = image_file

    if duplicates:

        print(f"\nClass: {class_folder.name}")

        for duplicate in duplicates:
            print("Duplicate:", duplicate)