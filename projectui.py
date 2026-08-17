import base64
import json
import os
from pathlib import Path
import tempfile
import streamlit as st

from inference import load_model, predict_image

# PATHS & DYNAMIC METADATA LOADING
BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "adinkra_translation_key.json"
BG_IMAGE_PATH = BASE_DIR / "INTRO PROJECT_BG.jpeg"


def load_metadata(json_path=METADATA_PATH):
    """
    Dynamically load class names and symbol details from the JSON file.
    Safely handles both flat strings and nested dictionaries.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = {}
    symbol_info = {}

    for key, value in metadata.items():
        idx = int(key)

        # If the JSON just maps directly to a string name (e.g., "0": "Aban")
        if isinstance(value, str):
            name = value
            class_names[idx] = name

        # If the JSON is a dictionary with names and meanings (e.g., "0": {"name": "Aban"})
        elif isinstance(value, dict):
            name = value.get("name", f"Class {idx}")
            class_names[idx] = name

            meaning = value.get("meaning", "")
            if meaning:
                symbol_info[name] = {"meaning": meaning}

    return class_names, symbol_info


# Dynamically populate class names and meanings from JSON
CLASS_NAMES, SYMBOL_INFO = load_metadata()


@st.cache_resource
def load_model_cached():
    """
    Cache and load model dynamically using the total count from metadata.
    """
    num_classes = len(CLASS_NAMES)
    try:
        return load_model(num_classes=num_classes)
    except TypeError:
        # Fallback if load_model() in inference.py is configured without parameters
        return load_model()


def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Encode background image safely using absolute path resolution
if BG_IMAGE_PATH.exists():
    background_image_path = get_base64(BG_IMAGE_PATH)
else:
    background_image_path = ""

st.markdown(
    f"""
    <style>

    .stApp {{
        min-height: 100vh;
        background-image: 
        linear-gradient(
            rgba(250, 243, 224, 0.65),
            rgba(250, 243, 224, 0.65)
        ),
        url("data:image/jpeg;base64,{background_image_path}");

    }}


    /* Title styling */
    h1 {{
        color: #D4AF37;
        font-family: Georgia, serif;
        text-align: center;
    }}


    /* Subtitle */
    h2, h3 {{
        color: black;
        font-family: Georgia, serif;
    }}


    /* Normal text */
    p {{
        color: black;
        font-size: 18px;
    }}


    /* Buttons */
    .stButton button {{
        background-color: #FFD700;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }}


    .stButton button:hover {{
        background-color: #B8860B;
        color: white;
    }}


    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: rgba(250, 243, 224, 0.15);
        border-radius: 15px;
        padding: 15px;
    }}


    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Center the content */
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("𖤓 AdinkraViz")
st.subheader("Explore Akan Heritage Through Adinkra Symbols.")
st.write(
    """  **Where artificial intelligence meets Akan heritage.**

    Welcome to AdinkraViz! Upload an image to uncover its meaning, history, and cultural significance.
    """
)

# Store uploaded image in session state (fixed state key consistency)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Save image to session state
if uploaded_image is not None:
    st.session_state.uploaded_image = uploaded_image
    st.image(
        uploaded_image,
        caption="Discover Akan heritage through Adinkra symbols",
        use_container_width=True,
    )

# Information section
if st.button("Learn More"):
    st.write(
        """
        Adinkra symbols are visual symbols that represent concepts or aphorisms, originating from the Akan people of Ghana and the Gyaman people of Côte d'Ivoire in West Africa. 
        They are used extensively in fabrics, pottery, logos, and advertising. 
        Each symbol has a unique meaning and conveys traditional wisdom, aspects of life, or the environment.
        
        The symbols are often used to express values, beliefs, and social norms. 
        For example, the "Sankofa" symbol represents the idea of learning from the past to build a better future. The "Gye Nyame" symbol signifies the supremacy of God. 
        These symbols are not only decorative but also serve as a means of communication and storytelling within the Akan culture.
        
        AdinkraViz aims to help users explore and understand these rich cultural symbols by allowing them to upload images and receive predictions about the Adinkra symbols they contain.
        """
    )

# Prediction button
if st.button("Predict"):
    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image before predicting.")

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(st.session_state.uploaded_image.getbuffer())
            temp_file_path = temp_file.name

        try:
            with st.spinner("Analyzing your Adinkra symbol..."):
                prediction = predict_image(load_model_cached(), temp_file_path)

            if isinstance(prediction, int):
                symbol_name = CLASS_NAMES.get(prediction, f"Symbol #{prediction}")
            else:
                symbol_name = str(prediction)

            st.success(f"Prediction: {symbol_name}")

            if symbol_name in SYMBOL_INFO:
                meaning = SYMBOL_INFO[symbol_name].get("meaning", "")

                if meaning:
                    st.info(f"**Meaning:** {meaning}")
            else:
                st.info("Information about this symbol is coming soon.")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
