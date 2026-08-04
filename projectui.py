import base64
import streamlit as st
import tempfile
import os
from inference import load_model, predict_image


@st.cache_resource
def load_model_cached():
    return load_model()


def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image_path = get_base64("INTRO PROJECT 1.png")


st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: 
        linear-gradient(
            rgba(250, 243, 224, 0.65),
            rgba(250, 243, 224, 0.65)
        ),
        url("data:image/png;base64,{background_image_path}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
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
        background-color: #D4AF37;
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


st.title("𓂀 AdinkraViz")
st.subheader("Explore Akan Heritage Through Adinkra Symbols.")
st.write(
    """ 🌿 **Where artificial intelligence meets Akan heritage.**

    Welcome to AdinkraViz! Upload an image to uncover its meaning, history, and cultural significance.
    """
)

# Store uploaded image in memory
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# save image
if uploaded_image:
    st.session_state.uploaded_image = uploaded_image
    st.image(
        uploaded_image,
        caption="Discover Akan heritage through Adinkra symbols",
        use_column_width=True,
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

        prediction = predict_image(load_model_cached(), temp_file_path)

        st.success(f"Prediction class ID: {prediction}")

        os.remove(temp_file_path)
