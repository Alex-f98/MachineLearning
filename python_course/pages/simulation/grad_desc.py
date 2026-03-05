import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from python_course.code.utils import Himmelblau, Polinomio1D
from python_course.code.numeric_solve import gradiente_descendiente


st.markdown("""
* **[Gradiente Descendente](#gradiente-descendente)**
* **[Complejizando la funcion de costo](#complejizando-la-funcion-de-costo)**
* **[Pseudocódigo](#pseudocodigo)**
* **[Visualización interactiva](#visualizacion-interactiva)**


""")

st.markdown("### Gradiente Descendente")
st.markdown(
r"""

    El método del gradiente descendente es un algoritmo numérico de optimización introducido por Cauchy hace muchos años, 
    y sin embargo sigue siendo la esencia de la mayoría de los algoritmos modernos de inteligencia artificial. 
    
    La idea es sencilla: igualar numéricamente a cero la derivada de una función a minimizar \(J(\theta)\). 
    
    Es decir, avanzar poco a poco (de forma iterativa) en la dirección del mayor decrecimiento de la función.

    $$
    \theta_{t+1} = \theta_t - \alpha \cdot \nabla J(\theta_t+1) \tag{2.11}
    $$

    donde $\alpha > 0$ recibe el nombre de *learning rate* o tasa de aprendizaje. 
    Este tipo de parámetros, que no se deciden durante el entrenamiento, reciben el nombre de hiperparámetros, 
    para diferenciarlos de los parámetros entrenables. 
    Según el valor de $\alpha$, el comportamiento del algoritmo puede ser muy distinto. 
    
 """
)

# Parámetros iniciales
max_iter, w_0 = 100, 2.4
learning_rates = [0.14, 0.0005, 0.07]

# Crear pestañas
tab_convexo, tab_no_convexo = st.tabs(["Convexo", "No Convexo"])

def plot_tab(poli, w_0, it, learning_rates):
    cols = st.columns(len(learning_rates))
    for i, lr in enumerate(learning_rates):
        with cols[i]:
            st.markdown(f'### Learning Rate: {lr}')
            fig = poli.dinamic_plot_2d_polinomio_with_trajectory(start_w=w_0, it=it, lr=lr)
            st.pyplot(fig)
    
# --- Tab Convexo ---
with tab_convexo:
    # Slider aparece arriba, junto al título de la pestaña
    it_c = st.slider("Iteraciones Convexo", 0, max_iter, 10)
    poli_c = Polinomio1D(max_iter=max_iter, convex=True)
    plot_tab(poli_c, w_0, it_c, learning_rates)

# --- Tab No Convexo ---
with tab_no_convexo:
    # Slider aparece arriba, junto al título de la pestaña
    it_nc = st.slider("Iteraciones No Convexo", 0, max_iter, 10)
    poli_nc = Polinomio1D(max_iter=max_iter, convex=False)
    plot_tab(poli_nc, w_0, it_nc, learning_rates)

st.markdown(
    r"""
    Se puede observar un ejemplo convergente del algoritmo. 
    Paso a paso el algoritmo se va acercando al mínimo, aunque corre el riesgo de necesitar muchas iteraciones para alcanzar la convergencia. 
    
    Sin embargo un learning rate muy grande, lejos de acelerar, puede generar comportamientos
    divergentes en el algoritmo.

    Por desgracia no existe un optimizador universal que funcione para cualquier tarea y
    conjunto de datos, dependerá de en cada problema el encontrar un valor de $\alpha$ adecuado.

    En la práctica se suele apuntar al valor más grande que genere un comportamiento
    convergente, eligiendo por prueba y error.  
    """
)

st.markdown("---")
st.markdown("### Complejizando la funcion de costo ")
st.markdown(
    """
    Vimos en los gráficos anteriores como el gradiente descendente puede converger a un mínimo global o local según si es convexa o no convexa respectivamente.
    
    Por lo general, las funciones de costo no suelen ser tan sencillas como los polinomios que vimos anteriormente y mucho menos convexas para modelos complejos, 
    por lo que es importante ganar un poco más de intuición sobre cómo funciona el gradiente descendente en superficies de costo más complejas.
    """
)

st.markdown("**Definición de la función Himmelblau**")
st.markdown(
    """
    La función Himmelblau es una función de dos variables que es una función no convexa con 4 mínimos locales.
    
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
    por tener 4 mínimos locales en 
    * $(3.0, 2.0)$
    * $(-2.805118, 3.131312)$
    * $(-3.779310, -3.283186)$
    * $(3.584428, -1.848126)$

    Dependiendo de la inicialización, el gradiente descendente puede converger a un mínimo local o global.
    Puedes notar que dependiendo de la inicialización podrá caer hacia un mínimo y no tener el *lr* suficiente para escapar de ese mínimo.
    A su vez teniendo un lr muy pequeño, el algoritmo puede tomar muchas iteraciones para converger pero con un lr muy grande puede no converger a ningún mínimo.
    """
)

####-----------------------------------------------------------------------------------###
#                                       Pseudocódigo y simulación
####-----------------------------------------------------------------------------------###
st.markdown("### Pseudocodigo")
st.info(
    r"""
    **Pseudocódigo: Gradiente Descendente**

    Sea una función objetivo $J(\theta)$ con gradiente $\nabla J(\theta)$.

    **Entrada:**
    - $ \theta_0 $: vector inicial
    - $\nabla J(\theta)$: función que calcula el gradiente
    - $\alpha$: tasa de aprendizaje (learning rate)
    - $\text{max\_iter}$: número máximo de iteraciones
    - $\varepsilon$: tolerancia para detener el algoritmo

    ---

    **Algoritmo:**

    1. Inicializar  
    
    $$
    \theta \leftarrow \theta_0
    $$

    Guardar $ \theta $ en la lista de pasos.

    2. Para $ i = 1 $ hasta $ \text{max\_iter} $:  
        * a. Calcular el gradiente:  

        $$
        g \leftarrow \nabla J(\theta)
        $$

        * b. Actualizar el vector de parámetros:  
    
        $$
        \theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)
        $$

        * c. Si la norma del gradiente es menor que la tolerancia:  
    
        $$
        \| \nabla J(\theta) \| < \varepsilon \quad \Rightarrow \quad \text{detener}
        $$

        * d. Guardar $ \theta $ en la lista de pasos.

    3. Devolver la lista de pasos y el valor final de $\theta$.
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

    Como se vé el algoritmo requiere los gradientes de la funcion objetivo para poder calcular el gradiente descendente.
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

###--------------------------------------------------------------------###
##
##                        Visualización interactiva
##
###--------------------------------------------------------------------###

st.markdown("---")

st.info(
    """
    Prueba con diferentes valores de valores de inicio y learning rate para ver cómo afectan 
    al gradiente descendente y a su convergencia.
    """
)

st.markdown("### Visualizacion interactiva")
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
