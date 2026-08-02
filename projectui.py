import base64

import streamlit as st

st.set_page_config(page_title="AdinkraViz", layout="wide")


# Function to convert image to base64
def get_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()


# Replace with your downloaded image
img = get_base64("background_adinkraviz.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Optional: Make the main container transparent */
    .block-container {{
        background: rgba(255, 255, 255, 0);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(255,255,255,0.9);
    }}

    /* Upload box */
    [data-testid="stFileUploader"] {{
        background-color: rgba(255,255,255,0.75);
        border-radius: 12px;
        padding: 10px;
    }}

    /* Buttons */
    .stButton>button {{
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center;'>◈✦◈ AdinkraViz</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Explore Akan Heritage Through Adinkra Symbols</h3>",
    unsafe_allow_html=True,
)
st.write(
    """
    **Welcome to AdinkraViz!**
    ✨ Where artificial intelligence meets cultural heritage.
    
    Upload an Adinkra symbol and uncover its meaning, history, 
    and cultural significance.
    """
)

uploaded_image = st.file_uploader(
    "Upload an Adinkra symbol image", type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Discover Akan heritage through Adinkra symbols.")

if st.button("Learn More"):
    st.write(
        "Adinkra symbols are visual symbols that represent concepts, originating from the Akan people of Ghana "
        "and the Gyaman people of Cote d'Ivoire in West Africa. "
        "They are used extensively in fabrics, pottery, logos, and advertising. "
        "Each symbol has a unique meaning and conveys traditional wisdom, aspects of life, or the environment."
    )

if st.button("Explore Symbols"):
    if uploaded_image:
        st.write("Running AI model to explore Adinkra symbols...")
    else:
        st.warning("Please upload an image to explore Adinkra symbols.")
