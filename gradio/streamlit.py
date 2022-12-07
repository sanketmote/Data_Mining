import streamlit
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd


streamlit.set_option("deprecation.showPyplotGlobalUse", False)



# Set all env variables
WCE_LOGO_PATH = "https://img.collegepravesh.com/2018/11/WCE-Sangli-Logo.png"

# wceLogo = Image.open(WCE_LOGO_PATH)

streamlit.set_page_config(
    page_title="Data Mining Project",
    page_icon=WCE_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
streamlit.markdown(hide_streamlit_style, unsafe_allow_html=True)


streamlit.markdown("<br />", unsafe_allow_html=True)

cols = streamlit.columns([2, 2, 8])

with cols[1]:
    streamlit.image(WCE_LOGO_PATH, use_column_width="auto")

with cols[2]:
    streamlit.markdown(
        """<h2 style='text-align: center; color: red'>Walchand College of Engineering, Sangli</h2>
<h6 style='text-align: center; color: white'>(An Autonomous Institute)</h6>""",
        unsafe_allow_html=True,
    )
    streamlit.markdown(
        "<h2 style='text-align: center; color: white'>DATA Mining ESE</h2>",
        unsafe_allow_html=True,
    )

# with cols[3]:
#     streamlit.image(wceLogo, use_column_width='auto')
streamlit.markdown("<hr />", unsafe_allow_html=True)
# streamlit.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)

styles = {
    "container": {
        "margin": "0px !important",
        "padding": "0!important",
        "align-items": "stretch",
        "background-color": "#fafafa",
    },
    "icon": {"color": "black", "font-size": "20px"},
    "nav-link": {
        "font-size": "20px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#eee",
    },
    "nav-link-selected": {
        "background-color": "lightblue",
        "font-size": "20px",
        "font-weight": "normal",
        "color": "black",
    },
}

with streamlit.sidebar:
    streamlit.markdown(
        """<h1>Welcome back,</h1>
    <h3>2019BTECS00113<br /></h3>""",
        unsafe_allow_html=True,
    )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)

    main_option = None
    dataframe = None

    main_option = option_menu(
            "",
            [
                "Data Analysis",
            ],
            icons=["clipboard-data", "eyeglasses"],
            default_index=0,
        )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)



if main_option == "Data Analysis":
    file = st.file_uploader("Upload Data Set")
    if(file):
        df = pd.read_csv(file)
        st.dataframe(df)
        st.write("Hello World!")
        


