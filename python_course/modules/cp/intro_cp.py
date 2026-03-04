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
st.image(
    "python_course/image/cp_img/cp_clf_reg.png",
    caption="Flujo de la predicción conforme, a izquierda se muestra el caso de clasificación y a la derecha el caso de regresión",
    use_container_width=True
)

st.markdown(
    r"""
    EL flujo principal de la predicción conforme es el siguiente:
    - Debemos partir de un modelo ya entrenado (o entrenarlo con su respectivo set de entrenamiento)
    - A partir del modelo se deben obtener predicciones ($\hat{y}_i$) y un score ($s_i \in \mathbb{R}$) por cada muestra de predicha.
    - Dicho par $(\hat{y}_i, s_i)$ se le conoce como el conjunto de calibracion (se obtuvo con un set de calibración $X_{test}$)
    - Se debe elejir un valor de confianza $\alpha$ (por ejemplo, 0.1 para 90% de confianza)
    - Luego se pasa por el algortimo de conformal prediction para generar los intervalos de predicción.
    - Estos intervalos de predicción deben cumplir para un nuevo punto $X_{test}$ con $\mathcal{P}(y_{test} \in \mathcal{C}(X_{t    est})) \geq 1 - \alpha$
    
    Osea que el marco de predicción conforme nos da una for
    """


)
