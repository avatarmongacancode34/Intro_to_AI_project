import base64
import streamlit as st
import tempfile
import os
from inference import load_model, predict_image

CLASS_NAMES = {
    0: "Aban",
    1: "Abe Dua",
    2: "Adinkrahene Dua",
    3: "Adinkrahene",
    4: "Adwera",
    5: "Adwo",
    6: "Agyinduwura",
    7: "Akoben",
    8: "Akofena",
    9: "Akokonan",
    10: "Akoma",
    11: "Akoma Ntoso",
    12: "Ananse Ntontan",
    13: "Ani Bere",
    14: "Asase Ye Duru",
    15: "Aya",
    16: "Bese Saka",
    17: "Bi Nnka Bi",
    18: "Biribi Wo Soro",
    19: "Boa Me Na Me Mmoa Wo",
    20: "Dama Dame",
    21: "Denkyem",
    22: "Dono",
    23: "Duafe",
    24: "Dwannimmen",
    25: "Eban",
    26: "Epa",
    27: "Ese Ne Tekrema",
    28: "Fafanto",
    29: "Fawohudie",
    30: "Fihankra", 
    31: "Fofo", 
    32: "Funtumfunafu Denkyem Funafu",
    33: "Gye Nyame",
    34: "Hwemudua",
    35: "Hye Won Hye",
    36: "Kae Me",
    37: "Kete Pa",
    38: "Kintinkantan",
    39: "Kojo Baiden",
    40: "Kontire Ne Akwamu",
    41: "Krado",
    42: "Kramo Bone",
    43: "Kuntinkantan",
    44: "Kwatakye Atiko",
    45: "Mako",
    46: "Mate Masie",
    47: "Mframadan",
    48: "Mmere Dane",
    49: "Mmomudwan",
    50: "Mmusuyidee",
    51: "Mpatapo",
    52: "Mpuannum",
    53: "Nea Onnim No Sua A Ohu",
    54: "Nea Ope Se Nkrofoo Ye Ma Wo No - Ye Saa Ara Ma Won",
    55: "Nea Ope Se Obedi Hene",
    56: "Nkonsonkonson",
    57: "Nkontim",
    58: "Nkuma Kese",
    59: "Nkyimu",
    60: "Nkyinkyim",
    61: "Nnonnowa",
    62: "Nsaa", 
    63: "Nserewa",
    64: "Nsoromma",
    65: "Nya Abotere",
    66: "Nyame Akruma",
    67: "Nyame Biribi Wo Soro",
    68: "Nyame Dua",
    69: "Nyame Nnwu Na Mawu",
    70: "Nyame Nti",
    71: "Nyame Ye Ohene",
    72: "Nyansapo",
    73: "Odo Nyera Fie Kwan",
    74: "Ohen Adwae",
    75: "Ohene",
    76: "Ohene Aniwa",
    77: "Ohene Tuo",
    78: "Okodee Mmowere",
    79: "Okuafo Pa",
    80: "Onyakopon Atom Nti Biribiara Beye Yie", 
    81: "Onyakopon Aniwa",
    82: "Onyakopon Ne Yen Ntena",
    83: "Osidan",
    84: "Osram",
    85: "Osram Ne Nsromma",
    86: "Ow Foro Adobe",
    87: "Owuo Atwedee",
    88: "Owuo Kum Nyame",
    89: "Pa Gya",
    90: "Sankofa",
    91: "Sepow",
    92: "Sesa Woruban",
    93: "Sunsum",
    94: "Tabon",
    95: "Tamfo Bebre",
    96: "Tumi Te Se Kosua",
    97: "Tuo Ne Akofena",
    98: "Wawa Aba",
    99: "Wo Nsa Da Mu A",
    100: "Wuforo Dua Pa"
}

SYMBOL_INFO = {

}

@st.cache_resource
def load_model_cached():
    return load_model()


def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image_path = get_base64("INTRO PROJECT_BG.jpeg")

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


st.title("𓂀 AdinkraViz")
st.subheader("Explore Akan Heritage Through Adinkra Symbols.")
st.write(
    """  **Where artificial intelligence meets Akan heritage.**

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

        with st.spinner("Analyzing the image..."):
            prediction = predict_image(load_model_cached(), temp_file_path)

        symbol_name = CLASS_NAMES[prediction]
        st.success(f"Prediction: {symbol_name}")
        st.write(f"Symbol Meaning: {SYMBOL_INFO[symbol_name]['meaning']}")

        os.remove(temp_file_path)
