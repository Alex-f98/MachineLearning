import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from python_course.code.utils import Himmelblau, Polinomio1D
from python_course.code.numeric_solve import gradiente_descendiente



st.markdown("### Polinomio 1D")

col1, col2, col3 = st.columns(3)
learning_rates = [0.14, 0.0005, 0.07]

for i, lr in enumerate(learning_rates):
    with eval(f'col{i+1}'):
        st.markdown(f'### Learning Rate: {lr}')
        poli = Polinomio1D(max_iter=200)
        fig = poli.plot_2d_polinomio_with_trajectory(start_w=2.4, lr=lr)
        st.pyplot(fig)

st.markdown("---")

st.markdown("""
Indice:

* [Gradiente descendent](#gradiente-descendent)
* [Pseudocódigo y simulación](#pseudocódigo-y-simulación)
* [Visualización interactiva](#visualización-interactiva)
---
""", unsafe_allow_html=True)

st.markdown("###  Gradiente descendente")

st.markdown("Definición de la función Himmelblau")
st.markdown(
    """
    $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$
    """
)
st.code(
    """
    def himmelblau(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    """
)


st.markdown("Visualización de la función de Himmelblau")

# Malla para graficar la superficie
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

f_himmelblau = Himmelblau(x,y)
fig = f_himmelblau.plot_3d_himmeblau()

# Mostrar gráfico en Streamlit
st.plotly_chart(fig)


####-----------------------------------------------------------------------------------###
#                                       Curvas de nivel
####-----------------------------------------------------------------------------------###
st.markdown("**Curvas de nivel**")

x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
f_himmelblau = Himmelblau(x,y)

fig = f_himmelblau.plot_2d_levels_himmeblau()
st.pyplot(fig)

st.markdown(
    """
    Como podemos observar, esta función es una función no-convexa caracterizada 
    por tener 4 mínimos locales en (3.0, 2.0), (-2.805118, 3.131312),
    (-3.779310, -3.283186) y (3.584428, -1.848126)
    """
)

####-----------------------------------------------------------------------------------###
#                                       Pseudocódigo y simulación
####-----------------------------------------------------------------------------------###
st.markdown("### Pseudocódigo y simulación")
st.info(
    r"""
    **Pseudocódigo: Gradiente Descendente**

    Sea una función objetivo $f(w)$ con gradiente $\nabla f(w)$.

    **Entrada:**
    - $ w_0 $: vector inicial
    - $\nabla f(w)$: función que calcula el gradiente
    - $\alpha$: tasa de aprendizaje (learning rate)
    - $\text{max\_iter}$: número máximo de iteraciones
    - $\varepsilon$: tolerancia para detener el algoritmo

    ---

    **Algoritmo:**

    1. Inicializar  
    
    $$
    w \leftarrow w_0
    $$

    Guardar $ w $ en la lista de pasos.

    2. Para $ i = 1 $ hasta $ \text{max\_iter} $:  
        * a. Calcular el gradiente:  

        $$
        g \leftarrow \nabla f(w)
        $$

        * b. Actualizar el vector de parámetros:  
    
        $$
        w \leftarrow w - \alpha \cdot g
        $$

        * c. Si la norma del gradiente es menor que la tolerancia:  
    
        $$
        \| g \| < \varepsilon \quad \Rightarrow \quad \text{detener}
        $$

        * d. Guardar $ w $ en la lista de pasos.

    3. Devolver la lista de pasos y el valor final de $w$.
    """
)

st.markdown("""
    El codigo que implementa el gradiente descendente es el siguiente:
    
    ```python
    # Gradient descent
    def gradiente_descendiente(w0, df, lr, max_iter, tol = 1e-6):
        w = np.array(w0, dtype=np.float64)
        steps = [w.copy()]
        for i in range(max_iter):
            grad = df(w)
            w -= lr * grad

            if np.linalg.norm(grad) < tol:
              break

            steps.append(w.copy())

        return np.array(steps), w
    ```

    Como se vé el algortimo requiere los gradientes de la funcion objetivo para poder calcular el gradiente descendente.
    ```python
    def grad_himmelblau(x, y):
      df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
      df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
      return np.array([df_dx, df_dy])
    """
)

st.markdown(
    """
    Vamos a ver que pasa si usamos el gradiente descendente para encontrar el mínimo de la función Himmelblau.
    ```python
    steps, w = gradiente_descendiente([0.1, 0.2], df, 0.01, 100)
    print(f"Óptimo hallado en {w}")
    print(f"en {len(steps)} pasos")
    ```
    """
)

# Parámetros ajustables
lr       = st.slider("Tasa de aprendizaje (lr)", 0.001, 0.1, 0.01, step=0.001)
max_iter = st.number_input("Máx iteraciones", min_value=10, max_value=1000, value=100)
x0       = st.number_input("Valor inicial x", -6.0, 6.0, 0.1)
y0       = st.number_input("Valor inicial y", -6.0, 6.0, 0.2)
 
# Botón para correr el algoritmo
if st.button("Correr gradiente descendente"):  
    for lr in [0.14, 0.0005, 0.07]:
        poli = Polinomio1D(max_iter=200)
        trajectory, w_final = poli.plot_2d_polinomio_with_trajectory(start_w=2.4, lr=lr)
        for step in trajectory:
            fig = poli.plot_2d_polinomio_with_trajectory(start_w=step, lr=lr)
            st.pyplot(fig)
            st.write(f'Current weight: {step}')  
        st.success(f'Final weight for lr={lr}: {w_final}')

###--------------------------------------------------------------------###
##
##                        Visualización interactiva
##
###--------------------------------------------------------------------###

st.markdown("---")

st.markdown("### Visualización interactiva")
st.markdown("Ajusta los parámetros para ver cómo funciona el gradiente descendente:")

# Columnas desproporcionadas
col_params, col_viz = st.columns([1, 2])

with col_params:
    st.markdown("#### Parámetros")
    start_x = st.slider("X1", -4.0, 4.0, 0.0, step=0.1)
    start_y = st.slider("X2", -4.0, 4.0, 0.0, step=0.1)
    lr_viz  = st.slider("LR", 0.001, 0.1, 0.01, step=0.001)

with col_viz:
    f_himmelblau = Himmelblau()
    fig = f_himmelblau.plot_2d_himmelblau_with_trajectory(start_x, start_y, lr_viz)
    st.pyplot(fig)

st.markdown("---")





hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 