import streamlit as st

st.set_page_config(layout='wide')

st.title("Visualización de ML")

st.markdown("""
Esta página te llevará a una aplicación externa de visualización de Machine Learning.
""")


st.link_button(
    "Visualización de Machine Learning-clasificación", 
    "https://machine-learning-visualizacion.streamlit.app/",
    type="secondary"
)
