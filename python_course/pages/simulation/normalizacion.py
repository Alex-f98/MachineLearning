import streamlit as st
from python_course.code.utils import NormalizationExperiments

#
#
#=================================================================================#
#
#  Normalizacion
#
#=================================================================================#

exp = NormalizationExperiments()
 
# Parámetros configurables (sin hardcode)
params = {
    'w1_optimo': -10,
    'w2_optimo': -10,
    'scale1': 2,
    'scale2': 2
}
 
# Ejecutar experimento completo (ahora retorna una sola figura con 6 gráficos)
@st.cache_data
def run_experiment():
    return exp.run_experiment(**params)

X, y, fig_data, fig_3d, fig_2d = run_experiment()

st.markdown("""
Indice:

* [Normalizacion](#Normalizacion)
* [Intuición](#Intuición)
---
""", unsafe_allow_html=True)

st.markdown("### Normalización")
st.markdown(
    """
    La normalización es una de las transformaciones más críticas que se deben aplicar a los datos antes de alimentarlos a un algoritmo de aprendizaje automático. 
    Su objetivo principal es asegurar que todos los atributos numéricos tengan una escala similar, lo cual es vital porque la mayoría de los algoritmos 
    no funcionan bien cuando las características de entrada tienen rangos muy distintos.
    
    Existen diferentes enfoques y técnicas de normalización que se aplican en distintas etapas del desarrollo de un modelo, aquí se verá la normalización como preprocesamiento:

    Basicamente, dado un conjunto de datos de entrenamiento, se calculan los parámetros de normalización (media y desviación estándar) y se aplican a los datos de entrenamiento y prueba.
    """
)
st.latex(r"""
    x_i \in X_{test} \\
    \begin{align}
    \mu &= \frac{1}{n}\sum_{i=1}^{n}x_i \\
    \sigma &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}
    \end{align}
    """)

st.markdown("Estos parametros se usan para normalizar los datos de entrenamiento y prueba (tambien los de validación):")

st.latex(r"""
    \\
    \begin{align}
    x_i    &= \frac{x_i - \mu}{\sigma}   \text{ tal que }  x_i \in X_{train} \\
    x_{i'} &= \frac{x_{i'} - \mu}{\sigma} \text{ tal que } x_{i'} \in X_{test}
    \end{align}
    """)    

st.markdown(
    """
    Ahora bien, para la función de costo $J(W) = || XW - y ||^2$, donde $W$ es el vector de parámetros del modelo $(w_1, w_2, \ldots, w_n)$.
    
    Si normalizamos $X$ entonces estamos modificando la funcion de costo, ya que la matriz $X$ afecta directamente el valor de la funcion de costo.
    
    Esto produce que la funcioń de costo original  vuelva esfericas las superficies de nivel del costo usado y esto produce que el gradiente converja con mayor facilidad.
    """
)

st.image("python_course/image/costo2norm.png")

st.markdown(
    """
    La normalización es fundamental para algoritmos que utilizan el gradiente descendente. 
    Sin ella, la función de costo puede tener la forma de un "tazón alargado", lo que obliga al algoritmo a oscilar erráticamente y tardar mucho más tiempo 
    en converger al mínimo global.
    Al normalizar, la superficie de error se vuelve más esférica, permitiendo que el gradiente avance directamente hacia el mínimo.
    """
)

st.markdown("### Intuición")
st.markdown(
    """
    Bien, pero qué sucede paso a paso al normalizar? Tomemos un set de datos simple el cual decidimos normalizar.
    En el eje de las muestras $X_1$ y $X_2$ podemos ver que la normalización solo es un escalado de los datos,
    es decir, no estamos modificando la relación entre las variables, solo estamos modificando la escala de las variables.
    """
)

#st.plotly_chart(fig_data, theme=None)
st.pyplot(fig_data)

st.markdown(
    """
    Pero es como ya presentamos, es más que eso, tiene que afectar a la función de costo.

    Para este set de datos se le calcula la función de costo, la cual será el riesgo empírico $J(W) = || XW - y ||^2$ en donde se modelan los datos
    con una función lineal $\hat{y} = XW$, entonces dicha función será convexa en el espacio de los parámetros $w_1, w_2, \ldots, w_n$ (en este caso n=2).
    """
)
st.plotly_chart(fig_3d)

st.markdown(
    """
    Puedes jugar con las representaciones en 3D de la función de costo, a derecha puedes ver que la superficie 
    de error es esférica, mientras que a la izquierda la superficie de error es alargada.

    Bien, con el gráfico 3D uno pensaría que en la izquierda convergería más rápido al mínimo global, pero no es así, podemos pensar en la expresión del gradiente y su algoritmo de aprendizaje:
    """
    )
st.latex(r"""
    \nabla J(W) = X^T(XW - y)
    """)

st.latex(r"""
    W_{t+1} = W_t - \alpha \nabla J(W_t)
    """)

st.markdown(
    """
    > Donde $X^T$ es la matriz de features transpuesta, $XW$ es el vector de predicciones y $y$ es el vector de etiquetas.

    Puedes notar que el gradiente no es más que la pendiente de la función de costo en un punto dado, y el algoritmo de aprendizaje 
    actualiza los parámetros del modelo en la dirección opuesta a la pendiente.

    En dicho punto con la función de costo a izquierda es más pronunciada en cierta dirección que en otra, lo que significa que el gradiente es más grande en esa dirección 
    y por lo tanto el algoritmo de aprendizaje actualiza los parámetros del modelo en una mayor magnitud pero en la otra dirección
    la magnitud es menor.
    lo que puede llevar a un camino serpenteando y subóptimo e incluso oscilaciones alrededor del mínimo, 
    forzando a elegir un learning rate $\\alpha$ exageradamente pequeño para compensar ese comportamiento y llegar a la convergencia.

    Por otro lado, en la función de costo a la derecha, el gradiente es más pequeño y con una magnitud más controlada en cada dirección, 
    por lo tanto el algoritmo de aprendizaje actualiza los parámetros del modelo en una menor magnitud (escalado con el learning rate $\\alpha$), 
    lo que permite una convergencia más estable.
    """
)
st.pyplot(fig_2d)

st.markdown(
    """
    Finalmente, es importante tener en cuenta que el $w_{optimo}$ al que se llega luego de normalizar no es el mismo que el $w_{optimo}$ 
    al que se llega sin normalizar, pero ambos son válidos.

    Los parámetros del modelo $w$ se ajustan para que el modelo se adapte lo mejor posible a los datos.

    Si los atributos de entrada cambian de escala (por ejemplo, mediante una transformación lineal como la normalización o estandarización), 
    los pesos deben transformarse de manera inversa para que las predicciones sigan siendo las mismas (en este caso $w' = w \cdot \\text{scale}$).
    """
)

st.info(
    """
    Nota: 

    En esta explicación utilizamos como estimador $\hat{y}$ a una función lineal, no necesariamente tiene que serlo.

    De hecho cuando uno trabaja con redes neuronales, el estimador suele ser una función no lineal $\hat{y} = \phi(X)$ y normalizar sirve de igual manera!!!
    puesto que ayuda en la convergencia, evita el "exploding gradient" o sea que el gradiente explote, que sea muy grande.
    
    """
)




hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 