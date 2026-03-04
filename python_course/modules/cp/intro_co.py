import streamlit as st

st.markdown("""
Indice:

* [Introducción](#introduccion)
* [seccion 2](#section-2)
* [Bibliografía](#bibliografia)
---
""", unsafe_allow_html=True)

st.markdown("### Introducción")
st.markdown(
    """
    La **predicción conforme** (conformal prediction, también conocida como inferencia conforme) es un paradigma fácil de usar para crear conjuntos 
    o intervalos de incertidumbre estadísticamente rigurosos para las predicciones de dichos modelos.

    De manera crucial, estos conjuntos son válidos en un sentido libre de distribución:
    poseen *garantías explícitas* y no asintóticas incluso sin asumir una distribución de los datos **ni supuestos sobre el modelo**. 

    Es posible utilizar predicción conforme con cualquier modelo previamente entrenado, como una red neuronal, para producir conjuntos 
    que garanticen contener el valor real con una probabilidad especificada por el usuario, por ejemplo, del 90 %
    """
)
st.image("python_course/image/cp_img/cp_clf_reg.png")


st.markdown("### seccion 2")
st.markdown(
    """
    Contenido de la seccion
    """
)

st.markdown("### Bibliografía")
st.markdown(
    """
    [1] Anastasios N. Angelopoulos and Stephen Bates. (2022). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification
    """
)



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 