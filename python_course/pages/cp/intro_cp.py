import streamlit as st
from python_course.code.cp_examples import run
import numpy as np

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
st.info(
    """
    Osea que el marco de predicción conforme nos devulve conjuntos (a izquirda de la figura) o intervalos (a derecha) de **predicción
    con garantias estadisticas rigurosas** sobre cualquier modelo de aprendizaje automático.
    """
)

st.markdown(
    r"""
    Los pasos de conformal prediction se puede ver macroscopicamente del a siguiente manera:
    - Debemos partir de un modelo ya entrenado (o entrenarlo con su respectivo set de entrenamiento)
    - A partir del modelo se deben obtener predicciones ($\hat{y}_i$) y un score ($s_i \in \mathbb{R}$) por cada muestra de predicha.
    - Dicho par $(\hat{y}_i, s_i)$ se le conoce como el conjunto de calibracion (se obtuvo con un set de calibración $X_{test}$)
    - Se debe elejir un valor de significancia $\alpha$ (por ejemplo, 0.1 para 90% de confianza)
    - Luego se pasa por el algortimo de conformal prediction para generar los intervalos(o conjuntos) de predicción.
    - Estos intervalos de predicción deben cumplir para un nuevo punto $X_{test}$ con $\mathbf{P}(y_{test} \in \mathcal{C}(X_{t    est})) \geq 1 - \alpha$
    
    """
)

st.markdown("EL flujo es el siguiente:")
st.info(
    r"""

    1. Identificar la nocion de heuristica o incerteza del modelo previamente entrenado.
    2. Definir la funcion de puntuaje $s(x,y) \in \mathbf{R}$ tambien llamado funciones de no conformidad (Los puntajes mas grandes define un peor ajuste entre $x$ e $y$).
        - $s(x,y)$ pequeño → buen ajuste/compatibilidad.
        - $s(x,y)$ grande → mal ajuste/compatibilidad.
    3. Computa $\hat{q}$ como el $⌈(n+1)(1−α)⌉$-th quantile de la calibracion $s_1 = s(X_1, Y_1), ..., s_n = s(X_n, Y_n)$.

        En otras palabras, se calcula un umbral calibrado a partir de los datos de calibración:

        $\hat{𝑞}=quantile(s_1,…,s_n;\frac{⌈(𝑛+1)(1−𝛼)⌉}{𝑛})$ esto fija el tamaño del conjunto de predicción de forma que garantice cobertura  $1-\alpha$

    4. Usa el quantile para formar los conjuntos de predicción para nuevos ejemplos $X_{test}$:

        $C(X_{test}) = \{y : s(X_{test}, y) \leq \hat{q}\}$

        Es decir: todos los $y$ que no son “demasiado improbables” según el umbral $\hat{q}$


    """
)
#Quiero  que si le doy click a la seccion se despliege el texto oculto
with st.expander("**Teorema D.1 (Garantía de cobertura de la calibración conforme).**"):
    st.markdown(
        r"""

        Supongamos que $(X_i, Y_i)_{i=1,\dots,n}$ y $(X_\text{test}, Y_\text{test})$ son i.i.d. Entonces definimos $\hat q$ como

        $$
        \hat q = \inf \Bigg\{ q : \frac{|\{i : s(X_i, Y_i) \leq q\}|}{n} \;\ge\; \frac{\lceil (n + 1)(1 - \alpha)\rceil}{n} \Bigg\}.
        $$

        y los conjuntos de predicción resultantes como

        $$
        C(X) = \{y : s(X, y) \leq \hat q\}.
        $$

        Entonces,

        $$
        \mathbb{P}\big(Y_\text{test} \in C(X_\text{test})\big) \;\ge\; 1 - \alpha.
        $$

        Esta es la misma propiedad de cobertura que la ecuación (1) en la introducción, pero escrita de forma más formal.
        Como observación técnica, el teorema también se cumple si las observaciones satisfacen la condición más débil de **intercambiabilidad** (*exchangeability*); ver \[1].
        A continuación, probamos la cota inferior.

        ---

        **Prueba del Teorema 1.**

        Sea $s_i = s(X_i, Y_i)$ para $i=1,\dots,n$ y $s_\text{test} = s(X_\text{test}, Y_\text{test})$.
        Para evitar manejar empates, consideramos el caso en el cual los $s_i$ son distintos con probabilidad 1. Ver \[25] para una prueba en el caso general.

        Sin pérdida de generalidad, asumimos que los puntajes de calibración están ordenados de modo que

        $$
        s_{(1)} < \cdots < s_{(n)}.
        $$

        En este caso, tenemos que

        $$
        \hat q =
        \begin{cases}
            s_{(\lceil (n+1)(1-\alpha)\rceil)} & \text{si } \alpha \ge \frac{1}{n+1}\\
            \infty & \text{e.o.c}
            \end{cases}
        $$

        Notemos que, en el caso $\hat q = \infty$, se cumple $C(X_\text{test}) = \mathcal{Y}$, por lo que la propiedad de cobertura se satisface trivialmente; así que solo debemos tratar el caso $\alpha \ge 1/(n+1)$.

        Procedemos observando la igualdad de los dos eventos:

        $$
        \{Y_\text{test} \in C(X_\text{test})\} = \{s_\text{test} \leq \hat q\}.
        $$

        Combinando esto con la definición de $\hat q$, obtenemos:

        $$
        \{Y_\text{test} \in C(X_\text{test})\} = \{s_\text{test} \leq s_{(\lceil (n+1)(1-\alpha)\rceil)}\}.
        $$

        Aquí viene la idea crucial.
        Por la intercambiabilidad de las variables $(X_1,Y_1), \dots, (X_\text{test},Y_\text{test})$, tenemos:

        $$
        \mathbb{P}(s_\text{test} \leq s_{(k)}) = \frac{k}{n+1}, \quad \text{para todo entero } k.
        $$

        En palabras, $s_\text{test}$ es igualmente probable de caer en cualquier posición entre los puntos de calibración $s_{(1)},\dots,s_{(n)}$.
        Nótese que aquí la aleatoriedad es sobre todas las variables $s_1,\dots,s_n,s_\text{test}$.

        De aquí concluimos:

        > **Nota:**  
        > En la prueba se toma $k=\lceil (n+1)(1-\alpha)\rceil$. Entonces  
        >
        > $$
        > \mathbb{P}\big(s_{\text{test}}\le s_{(k)}\big)=\frac{k}{n+1}
        > =\frac{\lceil (n+1)(1-\alpha)\rceil}{n+1}\ge\frac{(n+1)(1-\alpha)}{n+1}=1-\alpha.
        > $$
        >
        > De ahí se obtiene la cota inferior de cobertura:  
        >
        > $$
        > \mathbb{P}(Y_{\text{test}}\in C(X_{\text{test}}))=\mathbb{P}(s_{\text{test}}\le\hat q)
        > \ge 1-\alpha.
        > $$

        ---

        Ahora la cota superior.
        Técnicamente, la cota superior solo se cumple cuando la distribución del score conforme es continua, evitando empates.
        En la práctica, sin embargo, esta condición no es importante, porque el usuario siempre puede añadir una cantidad ínfima de ruido aleatorio al score.

        ---

        **Teorema D.2 (Cota superior de la calibración conforme).**
        
        Adicionalmente, si los puntajes $s_1,\dots,s_n$ tienen una distribución conjunta continua, entonces

        $$
        \mathbb{P}\big(Y_\text{test} \in C(X_\text{test}, U_\text{test}, \hat q)\big) \;\le\; 1 - \alpha + \frac{1}{n+1}.
        $$

        """
    )


st.markdown(

    r"""
     ## Intuición del texto

    - Si el score $s(x,y)$ refleja bien la “dificultad” del modelo:

        - Para ejemplos “fáciles” (donde el modelo predice con poca incertidumbre), los valores $s$ son bajos → el cuantil elegido  $\hat{𝑞}$ no tiene que ser tan grande → los conjuntos $C(X)$ terminan siendo más pequeños y más precisos.

        - Para ejemplos “difíciles” (alta incertidumbre), los valores $s$ crecen → se necesitan intervalos más anchos para cubrir bien → los conjuntos  $C(X)$ se vuelven más grandes.

    - Si el score es “malo” (ejemplo: puro ruido aleatorio sin relación con la calidad de predicción):

        - Entonces los conjuntos que arma conformal prediction son básicamente aleatorios.

        - Pero como la garantía de cobertura se basa solo en el orden de los scores y no en su significado, igual se cumple la cobertura $1-\alpha$.

        - El problema es que esos conjuntos serán muy grandes e inútiles (no informan nada útil, aunque estadísticamente sean correctos).
        
    """
)




#results, figures = run(model_type="mlp", alpha=0.05)
#
#if results:
#    st.success("Modelo entrenado exitosamente")
#    st.json(results)
#    
#    if fig := figures.get_figure("predictions"):
#        st.pyplot(fig)




noise_level = st.slider(
    "Nivel de Dificultad (Ruido)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    format="%.1f"
)

# Checkbox para modo caché
use_cache = st.checkbox(
    "🔒 Usar datos de referencia (cache)",
    help="Activa para usar siempre las mismas imágenes y poder comparar efectos de α y ruido consistentemente"
)

# Botón para limpiar caché (solo si está activo)
if use_cache:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Limpiar Caché", help="Reinicia los índices cacheados"):
            from python_course.code.cp_examples import cache_manager
            cache_manager.clear_cache()
            st.success("✅ Caché limpiado - Se seleccionarán nuevas imágenes")
            st.rerun()

# Mostrar información del caché si está activo
if use_cache:
    from python_course.code.cp_examples import cache_manager
    if cache_manager.is_cached:
        st.info(f"🔒 **Modo Cache Activado** - Índices guardados: {cache_manager.get_visualization_indices()}")
    else:
        st.info("🔒 **Modo Cache Activado** - Se guardarán los índices de la primera visualización")
else:
    st.info("🔄 **Modo Dinámico** - Se generarán nuevos datos aleatorios en cada ejecución")

st.markdown("""
## 📊 Ajusta el nivel de confianza (α)
##  $1 - \hat{f}(X_i)_{Y_i} = S \leq q$ entonces $\hat{f}(X_i)_{Y_i} \geq 1-q$
""")

# Valores logarítmicos entre 0 y 1
alpha_options = np.logspace(-3, -0.01, 40)  # De 0.001 a ~0.977 (evitar valores > 1)
alpha = st.select_slider(
    "Nivel de Significancia (α)",
    options=alpha_options,
    value=alpha_options[np.argmin(np.abs(alpha_options - 0.1))],
    format_func=lambda x: f"{x:.3f}"
)

#if st.button("🚀 Ejecutar Predicción", type="primary"):
with st.spinner("Entrenando modelo..."):
    results1, figures1 = run(model_type="mlp", alpha=alpha, noise_level=noise_level, use_cache=use_cache)
    
    if results1:
        st.success("✅ Modelo entrenado exitosamente!")
        
        # Métricas principales
        col0, col1, col2, col3, col4, col5 = st.columns(6)

        with col0:
            st.metric("*Nivel de Confianza:**", f"{(1 - alpha):.3f}")
        
        with col1:
            st.metric("Accuracy", f"{results1['accuracy']:.4f}")
        
        with col2:
            st.metric("Threshold (q)", f"{results1['threshold']:.4f}")
        
        with col3:
            st.metric("Umbral (1-q)", f"{1 - results1['threshold']:.4f}")
        
        with col4:
            st.metric("Modo", results1['cache_status'])
        
        with col5:
            modelo_status = "✅ Entrenado" if results1['model_trained'] else "🔄 Entrenando..."
            st.metric("Modelo", modelo_status)

        
        # Siempre mostrar el gráfico
        if fig := figures1.get_figure("predictions"):
            st.pyplot(fig)
    
    else:
        st.error("❌ Error en la ejecución")


st.markdown(
    """
    Bien, se puede jugar con la simulación para ganar intuición, se aumentamos el ruido en las muestras
    los sets, las predicciones se vuelven menos precisas, esto hace que los scores sean mas grandes.
    Si los scores crecen, entonces van a haber menos muestras que cumplan con la condición 
    $1 - \hat{f}(X_i)_{Y_i} = S \leq q$, 
    lo que hace que los conjuntos sean mas pequeños.
    """
)