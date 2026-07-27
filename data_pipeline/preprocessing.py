import cv2 as cv
import numpy as np
import os


# ==========================================
# 1. LOAD IMAGE
# ==========================================

def load_image(image_path):
    """
    Reads an image from a given path.

    OpenCV loads images as BGR format.
    """

    img = cv.imread(image_path)

    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    return img



# ==========================================
# 2. RESIZE IMAGE
# ==========================================

def resize_image(img, size=(224,224)):
    """
    Resizes image to the size required by CNN models.

    Default:
    224 x 224 pixels
    """

    resized = cv.resize(
        img,
        size,
        interpolation=cv.INTER_AREA
    )

    return resized



# ==========================================
# 3. CONVERT BGR TO RGB
# ==========================================

def convert_to_rgb(img):
    """
    Converts OpenCV BGR format into RGB format.

    Useful because most ML libraries use RGB.
    """

    rgb = cv.cvtColor(
        img,
        cv.COLOR_BGR2RGB
    )

    return rgb



# ==========================================
# 4. NORMALIZE IMAGE
# ==========================================

def normalize_image(img):
    """
    Converts pixel values from:

    0 - 255

    into:

    0 - 1
    """

    normalized = img / 255.0

    return normalized



# ==========================================
# 5. GAUSSIAN BLUR
# ==========================================

def blur_image(img):
    """
    Removes small noise from an image.

    Use carefully because too much blur
    can remove symbol details.
    """

    blurred = cv.GaussianBlur(
        img,
        (5,5),
        0
    )

    return blurred



# ==========================================
# 6. SHARPEN IMAGE
# ==========================================

def sharpen_image(img):
    """
    Enhances edges and details.
    """

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    sharpened = cv.filter2D(
        img,
        -1,
        kernel
    )

    return sharpened



# ==========================================
# 7. ROTATE IMAGE (DATA AUGMENTATION)
# ==========================================

def rotate_image(img, angle=10):
    """
    Rotates image slightly.

    Used for data augmentation.

    Example:
    -10 degrees
    +10 degrees
    """

    height, width = img.shape[:2]

    center = (
        width // 2,
        height // 2
    )


    rotation_matrix = cv.getRotationMatrix2D(
        center,
        angle,
        1.0
    )


    rotated = cv.warpAffine(
        img,
        rotation_matrix,
        (width,height)
    )


    return rotated



# ==========================================
# 8. BRIGHTNESS AUGMENTATION
# ==========================================

def change_brightness(img, value=30):
    """
    Changes image brightness.

    Useful because users may take photos
    under different lighting.
    """

    hsv = cv.cvtColor(
        img,
        cv.COLOR_RGB2HSV
    )

    hsv[:,:,2] = np.clip(
        hsv[:,:,2] + value,
        0,
        255
    )

    bright = cv.cvtColor(
        hsv,
        cv.COLOR_HSV2RGB
    )

    return bright



# ==========================================
# 9. COMPLETE PREPROCESSING PIPELINE
# ==========================================

def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.

    Steps:

    1. Load image
    2. Resize
    3. Convert BGR -> RGB
    4. Normalize

    Returns processed image ready for CNN.
    """


    img = load_image(image_path)


    if img is None:
        return None


    img = resize_image(img)


    img = convert_to_rgb(img)


    img = normalize_image(img)


    return img



# ==========================================
# 10. PROCESS COMPLETE DATASET
# ==========================================

def process_dataset(input_folder, output_folder):
    """
    Processes all images in a folder.

    Example:

    raw_dataset/
          gye_nyame/
              image1.jpg

    becomes:

    processed_dataset/
          gye_nyame/
              image1.jpg
    """


    os.makedirs(
        output_folder,
        exist_ok=True
    )


    valid_extensions = (".jpg", ".jpeg", ".png")


    for image_name in os.listdir(input_folder):

        if not image_name.lower().endswith(valid_extensions):
            print("Skipping:", image_name)
            continue

        image_path = os.path.join(
            input_folder,
            image_name
        )


        img = load_image(image_path)


        if img is None:
            continue


        img = resize_image(img)


        img = convert_to_rgb(img)


        # Convert back to uint8
        # before saving

        img = (img * 255).astype(
            np.uint8
        )


        save_path = os.path.join(
            output_folder,
            image_name
        )


        cv.imwrite(
            save_path,
            cv.cvtColor(
                img,
                cv.COLOR_RGB2BGR
            )
        )


        print(
            "Processed:",
            image_name
        )
# ==========================================
# PROCESS ALL ADINKRA SYMBOL CLASSES
# ==========================================

def process_all_symbols(raw_folder, processed_folder):
    """
    Processes every Adinkra symbol folder.

    Example:

    raw_data/
        nsoroma/
        sankofa/
        gye_nyame/

    becomes:

    processed_data/
        nsoroma/
        sankofa/
        gye_nyame/
    """


    # Get all symbol folders
    symbols = os.listdir(raw_folder)


    for symbol in symbols:

        input_folder = os.path.join(
            raw_folder,
            symbol
        )


        output_folder = os.path.join(
            processed_folder,
            symbol
        )


        # Ignore files that are not folders
        if not os.path.isdir(input_folder):
            continue


        print("\nProcessing symbol:", symbol)


        process_dataset(
            input_folder,
            output_folder
        )


    print("\nAll symbols processed successfully!")

def create_label_mapping(dataset_folder):

    symbols = os.listdir(dataset_folder)

    symbols = sorted(symbols)


    label_map = {}


    for index, symbol in enumerate(symbols):

        label_map[symbol] = index


    return label_map

def load_dataset(dataset_folder):

    label_map = create_label_mapping(dataset_folder)


    images = []
    labels = []


    for symbol in label_map:


        symbol_folder = os.path.join(
            dataset_folder,
            symbol
        )


        for image_name in os.listdir(symbol_folder):

            image_path = os.path.join(
                symbol_folder,
                image_name
            )


            images.append(image_path)

            labels.append(
                label_map[symbol]
            )


    return images, labels


# ==========================================
# TESTING SECTION
# ==========================================

if __name__ == "__main__":


    raw_folder = r"C:\Users\Vannessa Brose\OneDrive\Desktop\OpenCV\raw_data"


    processed_folder = r"C:\Users\Vannessa Brose\OneDrive\Desktop\OpenCV\processed_data"



    process_all_symbols(
        raw_folder,
        processed_folder
    )




# # ==========================================
# # TEST YOUR NSOROMA DATASET
# # ==========================================

# if __name__ == "__main__":


#     input_folder = r"C:\Users\Vannessa Brose\OneDrive\Desktop\OpenCV\raw_data\nsoroma"


#     output_folder = r"C:\Users\Vannessa Brose\OneDrive\Desktop\OpenCV\processed_data\nsoroma"



#     process_dataset(
#         input_folder,
#         output_folder
#     )


#     print("Dataset processing complete!")
       
