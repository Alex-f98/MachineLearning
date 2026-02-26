import streamlit as st

st.markdown("""
Indice:

* [seccion 1](#section-1)
* [seccion 2](#section-2)
* [seccion 3](#section-3)
---
""", unsafe_allow_html=True)

st.markdown("### seccion 1")
st.markdown(
    """
    Contenido de la seccion
    """
)

st.markdown("### seccion 2")
st.markdown(
    """
    Contenido de la seccion
    """
)

st.markdown("### seccion 3")
st.markdown(
    """
    Contenido de la seccion
    """
)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 