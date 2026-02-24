import streamlit as st
from st_pages import add_page_title, hide_pages

add_page_title(layout="wide")

hide_pages(["Thank you"])

st.markdown("""
Indice:

* seccion 1
* seccion 2
* seccion 3
---
""", unsafe_allow_html=True)

st.markdown("### seccion")
st.markdown("""
Contenido de la seccion
""")



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 