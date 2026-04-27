import streamlit as st
from python_course.code.cp_examples import run
import numpy as np
from python_course.code.cp_examples import StreamlitCache


cache = StreamlitCache()
PATH_IMAGES = "python_course/image/cp_img/"



st.markdown("""
Índice:

* [Introducción](#introduccion)
* [Fundamentos Teóricos](#fundamentos-teoricos)
  * [Teorema de cobertura](#teorema-de-cobertura)
  * [Intuición del método](#intuicion-del-metodo)
* [Simulación Interactiva](#simulacion-interactiva)
* [Conceptos Clave](#conceptos-clave)
  * [Definiciones importantes](#definiciones-importantes)
  * [Cálculo del umbral](#calculo-del-umbral)
* [Aplicaciones en Regresión](#aplicaciones-en-regresion)
  * [Regresión lineal y cuantílica](#regresion-lineal-y-cuantilica)
  * [Split Conformal Prediction](#split-conformal-prediction)
  * [Conformal Quantile Regression](#conformal-quantile-regression)
* [Comparación de Métodos](#comparacion-de-metodos)
* [Conclusiones](#conclusiones)
* [Bonus: Forecasting](#bonus-conformal-prediction-para-forecasting)

---""", unsafe_allow_html=True)

st.markdown("### Introducción")
st.markdown(
    """
    **¿Qué es la predicción conforme?**

    La Predicción conforme (o inferencia conforme) es un paradigma que utiliza la experiencia pasada para determinar niveles precisos de confianza en las predicciones futuras. Este no produce una predicción
    puntual $\hat{y}$ sino que produce una región de predicción (un conjunto o un intervalo) estadísticamente rigurosos.
    
    De manera crucial, estos conjuntos son válidos en un sentido libre de distribución:
    poseen *garantías explícitas* y no asintóticas incluso sin asumir una distribución de los datos **ni supuestos sobre el modelo**. 

    Es posible utilizar predicción conforme con cualquier modelo previamente entrenado (una black box), como una red neuronal, para producir conjuntos 
    que garanticen contener el valor real con una probabilidad especificada por el usuario, por ejemplo, del 90 %

    La predicción conforme (**CP**) proporciona una cobertura marginal, asegurando que la probabilidad de que la etiqueta verdadera y esté contenida en la región de predicción
    es de al menos $1 - \\alpha$, donde $\\alpha$ es el nivel de error elegido por el usuario. 
    
    > El término "marginal" significa que la probabilidad se calcula promediando (marginalizando) sobre toda la aleatoriedad inherente al proceso, lo que incluye tanto la elección de los datos del conjunto de calibración como la del nuevo punto de prueba, en la práctica, esto implica que si aplicas el método a una secuencia larga de predicciones nuevas, la frecuencia de aciertos a largo plazo será aproximadamente $1- \\alpha$ o superior.
    """
)

st.image(
    PATH_IMAGES + "cp_clf_reg.png",
    caption="Flujo de la predicción conforme, a izquierda se muestra el caso de clasificación y a la derecha el caso de regresión",
    width="stretch"
)
st.info(
    """
    O sea que el marco de predicción conforme nos devuelve conjuntos (a izquierda de la figura) o intervalos (a derecha) de **predicción
    con garantías estadísticas rigurosas** sobre cualquier modelo de aprendizaje automático.
    """
)

st.markdown(
    r"""
    Los pasos de conformal prediction se pueden ver macroscópicamente de la siguiente manera:
    - Debemos partir de un modelo ya entrenado (o entrenarlo con su respectivo set de entrenamiento)
    - A partir del modelo se deben obtener predicciones ($\hat{y}_i$) y un score ($s_i \in \mathbb{R}$) por cada muestra predicha.
    - Dicho par $(\hat{y}_i, s_i)$ se le conoce como el conjunto de calibración (se obtuvo con un set de calibración $X_{test}$)
    - Se debe elegir un valor de significancia $\alpha$ (por ejemplo, 0.1 para 90% de confianza)
    - Luego se pasa por el algoritmo de conformal prediction para generar los intervalos (o conjuntos) de predicción.
    - Estos intervalos de predicción deben cumplir para un nuevo punto $X_{test}$ con $\mathbf{P}(y_{test} \in \mathcal{C}(X_{test})) \geq 1 - \alpha$
    
    """
)

st.markdown("EL flujo es el siguiente:")
st.info(
    r"""

    1. Identificar la noción de heurística o incertidumbre del modelo previamente entrenado.
    2. Definir la función de puntuación $s(x,y) \in \mathbf{R}$ también llamado funciones de no conformidad (Los puntajes más grandes definen un peor ajuste entre $x$ e $y$).
        - $s(x,y)$ pequeño → buen ajuste/compatibilidad.
        - $s(x,y)$ grande → mal ajuste/compatibilidad.
    3. Computa $\hat{q}$ como el $⌈(n+1)(1−α)⌉$-th quantile de la calibración $s_1 = s(X_1, Y_1), ..., s_n = s(X_n, Y_n)$.

        En otras palabras, se calcula un umbral calibrado a partir de los datos de calibración:

        $\hat{𝑞}=quantile(s_1,…,s_n;\frac{⌈(𝑛+1)(1−𝛼)⌉}{𝑛})$ esto fija el tamaño del conjunto de predicción de forma que garantice cobertura  $1-\alpha$

    4. Usa el quantile para formar los conjuntos de predicción para nuevos ejemplos $X_{test}$:

        $C(X_{test}) = \{y : s(X_{test}, y) \leq \hat{q}\}$

        Es decir: todos los $y$ que no son “demasiado improbables” según el umbral $\hat{q}$


    """
)
st.markdown("## Fundamentos Teóricos")

### Teorema de cobertura

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
     ## Intuición del método

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



# Botón para limpiar caché (solo si está activo)

col1, col2 = st.columns([3, 1])
with col1:
    # Checkbox para modo caché
    use_cache = st.checkbox(
        "🔒 Usar datos de referencia (cache)",
        help="Activa para usar siempre las mismas imágenes y poder comparar efectos de α y ruido consistentemente"
    )
    cache.activate = use_cache
with col2:
    if st.button("🗑️ Limpiar Caché", help="Reinicia los índices cacheados"):
        cache.clear()
        st.success("✅ Caché limpiado - Se seleccionarán nuevas imágenes")
        st.rerun()

# Mostrar información del caché si está activo
if use_cache:
    if cache.keys():
        st.info(f"🔒 **Modo Cache Activado** - Índices guardados: {cache.get('indices')}")
    else:
        st.info("🔒 **Modo Cache Activado** - Se guardarán los índices de la primera visualización")
else:
    st.info("🔄 **Modo Dinámico** - Se generarán nuevos datos aleatorios en cada ejecución")

st.markdown("""
## Ajusta el nivel de confianza (α)
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
    results1, figures1 = run(model_type="mlp", alpha=alpha, noise_level=noise_level, cache=cache)
    
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
    Bien, se puede jugar con la simulación para ganar intuición:
    - Si aumentamos el ruido en las muestras (muestras más difíciles) desde luego que las predicciones del modelo se vuelven menos precisas (esto depende de la robustez del modelo por detrás).
    - Si el modelo es menos preciso, se traduce en scores más altos, esto hace que el threshold *q* también crezca.
    - Si los scores crecen, **q** también, luego $1 - \hat{f}(X_i)_{Y_i} = S \leq q$ resultará en conjuntos de predicción
    más grandes.
    - Al aumentar $\\alpha$ el umbral (1-q) aumenta, pudiendo disminuir las clases presentes en $C$.
    - Por lo anterior, aumentar el nivel de confianza (disminuir alpha) puede hacer que el conjunto de predicción sea más grande debido a que crece el umbral **q**.
    """
)

st.markdown("## Conceptos Clave")

st.markdown("### Definiciones importantes")

st.markdown(
    """
    **Algunas definiciones**

    - **Nivel de significancia (α):** Es la tasa de error que el usuario está dispuesto a aceptar (ej: 0.05 para un 95\% de confianza).

    - **Puntuación (Score) de no conformidad:** Es una función real que mide qué tan "inusual" o "diferente" se ve un nuevo ejemplo $X_{test}$ respecto a un conjunto de datos previos $X_{cal}$.
    Un score alto indica que el modelo está muy inseguro o que el dato es atípico.

    - **Intercambiabilidad (Exchangeability):** Es el único supuesto estadístico necesario. Indica que el orden de los datos no altera su distribución conjunta lo que permite que los 
    errores en el conjunto de calibración sean representativos de los errores futuros (También se suele usar el supuesto de IID que es más restrictivo).

    - **Validez vs eficacia:** La validez es la garantía de que el modelo acertará en el porcentaje prometido. La **eficacia** se refiere a que la región de predicción sea pequeña e informativa;
    esto último en principio depende de la calidad del modelo base utilizado.

    """
)

st.markdown("### Cálculo del umbral")

st.markdown(
    """
    En la siguiente imagen se tomaron 150 muestras de calibración y se calcularán sus respectivos scores a fin de calcular el umbral $\\hat{q}$.


    ```python
    alpha = 0.04

    n = len(X_calib_mnist)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    # q_val = 0.966

    # Tomar el score tal que al menos el 96% de los scores están por debajo
    q_hat = np.quantile(scores, q_val, method="higher")
    # q_hat = 0.784

    ```

    vemos que primero se decide el $\\alpha$, 
    luego se calcula el porcentaje (*q_val*)el cual indica el cuantil para **CP**, 
    con esos datos internamente se calcula el umbral **q** usando la funcion `np.quantile`.

    Internamente, dicha funcion toma los scores suministrados y los ordena.
    por otro lado calcula el indice del percentil dado q_val y el largo de los scores, con ello obtiene un score en dicho indice.
    
    Este score es el umbral **$\\hat{q}$**.
    > El argumento `higher` indica que si el indice no es un entero, se toma el valor superior de esta forma me aseguro la covertura de al menos $1-\\alpha$.
    
    
    """
)



st.image(
    PATH_IMAGES + "quantile_into.png",
    caption="Dentro de la funcion quantile, se puede ver como se calcula el umbral q",
    width="stretch"
)

st.markdown(
    """
    En la siguiente imagen se puede ver la forma mas intuitiva de calcular el umbral $\\hat{q}$: que es tomando el histograma de los
    scores, luego ir acumulando la suma hasta cubrir la probabilidad $1-\\alpha$ en ese punto se toma el valor del score alcanzado como el umbral $\\hat{q}$.
    
    > Este umbral calculado es meramente para intuición, cuanto mas grande sea el conjunto de calibración mejor.
    """
)

st.image(
    PATH_IMAGES + "histogram2_quantile.png",
    caption="Histograma de los scores, se calcula el percentil q para obtener el umbral",
    width="stretch"
)

st.markdown(
    """
    Una vez calculado el umbral $\\hat{q}$, se puede usar para predecir regiones de nuevos datos.
    """
)

st.markdown("## Aplicaciones en Regresión")

st.markdown("### Regresión lineal y cuantílica")    
st.markdown(
    """
    En regresión, el objetivo es predecir un valor **continuo** en lugar de una **clase**. 
    La idea de CP se puede extender a regresión de manera natural definiendo una noción de score de no conformidad.
    
    Se puede empezar con lo más simple, definiendo cuáles van a ser los datos de entrenamiento y testeo y obteniendo la recta de regresión lineal
    """
)
col1, col2 = st.columns(2)

with col1:
    st.image(PATH_IMAGES + "ols.png", caption="Regresión lineal", width="stretch")

with col2:
    st.image(PATH_IMAGES + "qr_test.png", caption="Regresión cuantílica", width="stretch")

st.markdown(
    """
    En la imagen de izquierda se presenta el conjunto de datos de entrenamiento y el de testeo con 201 y 132 puntos respectivamente.
    Los datos son generados de la siguiente forma:

    ```python
    n = 400
    X = np.random.uniform(low=0,high=5,size=n)
    sigma = 1

    ## Heteroscedastic
    y = np.cos(X) + (1-np.cos(X))*sigma*np.random.normal(size=n)
    # Luego se dividen los datos en entrenamiento y testeo
    ```
    En donde puedes ver que los datos se generan de forma que la varianza aumenta conforme x aumenta esto es para generar datos más realistas y heteroscedásticos.
    
    Además, se agregaron puntos outliers en ambos sets.

    A derecha se puede ver cómo se aplica la regresión cuantílica para predecir los intervalos de confianza.

    ## ¿Pero qué es la regresión cuantílica?

    Es similar a OLS, pero la función que minimiza es diferente, en lugar de minimizar la suma de cuadrados, minimiza la pinball loss de los errores.

    $$
    \\text{Risk}_{\\beta}(f) = \\mathbb{E}\\left[ \\text{pinball}_{\\beta}(Y, f(X)) \\right]
    $$

    donde $\\beta$ es el nivel del cuantil por ejemplo para este caso dará los niveles de confianza $\\beta/2=0.05$, $1 - \\beta/2 = 0.95$  si $\\beta = 0.1$.



    Es importante notar que en el cuantil 0.5 este no predice la media de y dado x, sino la mediana.

    > Nota: QR no tiene garantía teórica sobre la cobertura!

    Los resultados se pueden ver en la imagen de la izquierda que nos dan una estimación de los intervalos de confianza para diferentes niveles de confianza.

    metricas:

    | Métrica | Valor |
    |---------|-------|
    | Empirical Coverage (%) | 86.363636 |
    | Theorical Coverage (%) | 90.0 |
    | Sharpness | 3.826622 |
    | Winkler Score | 7.527342 |
    
    En resumen, la **regresión cuantílica** tiene como principal ventaja que ofrece una visión más completa de la relación entre las variables predictoras 
    y la variable respuesta, especialmente cuando la distribución condicional no es simétrica o cuando interesa analizar los **extremos (colas)** 
    de la distribución.

    Sin embargo, presenta algunas **limitaciones**: solo es aplicable a **variables de respuesta continuas**; la **interpretación de los resultados** 
    puede ser compleja, ya que cada cuantil representa una relación distinta; requiere un **tamaño muestral grande** para obtener estimaciones precisas

    """
)

st.markdown("### Split Conformal Prediction")
st.markdown(
    """
    Hasta ahora solo presentamos regresión lineal y cuantílica como el caso más simple de prediction intervals.
    Ahora, queda aplicar conformal prediction a estos métodos para obtener intervalos de confianza más robustos.

    Antes, no olvidar que se tiene que generar un set de calibración para obtener los q_{1-α/2} y q_{α/2}.
    Para ello se va a tomar un 25% de los datos de entrenamiento para el set de calibración.

    El primero y más intuitivo es el split conformal prediction, lo complicado es obtener esta noción de score que se daba naturalmente en el caso de clasificación.

    Aquí será: 

    $$
    S(x) = |f(x) - y|
    $$

    donde $f(x)$ es la predicción del modelo y $y$ es el valor real.
    
    y se obtendrán los cuantiles de los errores en el set de calibración para obtener los intervalos de confianza de la siguiente manera:

    $$
    C(X,y) = \{f(x) - q_{1-\\alpha/2}, f(x) + q_{1-\\alpha/2}\}
    $$

    los resultados se muestran a continuación:
    """
)


st.image(
    PATH_IMAGES + "scp_test.png",
    caption="Intervalos de confianza obtenidos con split conformal prediction",
    width="stretch"
)

st.markdown(
    """
    La **predicción conformal con partición (split conformal prediction)** ofrece una **garantía teórica de cobertura**, a diferencia de la regresión cuantílica. No obstante, presenta varias **limitaciones** importantes:

    *   **Dependencia de la partición de los datos**: el desempeño puede variar considerablemente según cómo se dividan los datos en entrenamiento y validación, generando intervalos distintos para diferentes particiones.
    *   **Supuesto de intercambiabilidad (generalmente i.i.d.)**: requiere que los datos sean intercambiables, lo cual no siempre se cumple en conjuntos con dependencia temporal, espacial u otras estructuras.
    *   **Uso ineficiente de los datos**: necesita un conjunto de validación separado, reduciendo los datos disponibles para entrenar el modelo, lo que es problemático cuando los datos son escasos o el modelo es complejo.
    *   **Intervalos no adaptativos al punto de prueba**: los intervalos producidos no se ajustan específicamente a cada punto de test, lo que puede resultar en predicciones menos informativas en contextos heterogéneos.

    Por lo tanto, se necesitan otros métodos que resuelvan estas limitaciones.
    """
)

st.markdown("### Conformal Quantile Regression")
st.markdown(
    """
    Si quiero una noción de scores desde la regresión cuantílica, puedo puntuar qué tan separados están mis cuantiles respecto al verdadero valor, y tomar siempre el de mayor distancia para tomar el peor caso.

    $$
    S(x,y) = \max \{ q_{\\alpha/2} - y, y - q_{1-\\alpha/2} \}
    $$

    Teniendo en cuenta que este score es una funcion de no conformidad, el cual es cero cuando las predicciones son buenas.

    Esto es un método distinto al SCP debido que provee de una predicción adaptativa a test

    El resultado es el siguiente:

    """
)
st.image(
    PATH_IMAGES + "qcr_test.png",
    caption="Intervalos de confianza obtenidos con conformal quantile regression",
    width="stretch"
)

st.markdown("## Comparación de Métodos")
st.markdown(
    """
    Como se puede ver, los intervalos de confianza obtenidos con conformal quantile regression son mucho más adaptativos a los puntos de test, 
    ya que se ajustan específicamente a cada punto de test.

    Se puede decir que son intervalos eficientes, heredado del método de QR.

    Las métricas totales son:

    | Method | Empirical coverage (%) | Theoretical coverage (%) | Sharpness | Winkler score |
    |--------|------------------------|--------------------------|-----------|---------------|
    | QR     | 86.36                  | 90.0                     | 3.83      | 7.53          |
    | SCP    | 92.42                  | 90.0                     | 6.11      | 9.93          |
    | CQR    | 94.70                  | 90.0                     | 4.35      | 7.64          |

    Como se puede ver, el método de CQR es el que tiene mejor sharpness, 
    lo que significa que los intervalos son más pequeños, pero a su vez tiene una cobertura empírica más baja que el método de SCP.

    Se puede ver mejor las metricas:
    """
)

st.image(
    PATH_IMAGES + "metrics_cp.png",
    caption="Métricas obtenidas con conformal prediction relativo a regresión cuantílica",
    width="stretch"
)

st.markdown(
    """
    Se puede concluir que el método de CQR es el que tiene mejor sharpness, 
    lo que significa que los intervalos son más pequeños, pero a su vez tiene una cobertura empírica más baja que el método de SCP.

    Por otro lado, esto fue analizado para un $\\alpha = 0.1$, pero se puede variar este parámetro para ver cómo afecta a las métricas y en particular me interesa cómo varía la cobertura empírica.
    """
)
st.image(
    PATH_IMAGES + "coverage_cp.png",
    caption="Cobertura empírica obtenida con conformal prediction",
    width="stretch"
)

st.markdown(
    """
    > Nota: [cqr_linear_model.ipynb](https://colab.research.google.com/drive/13CHq9cVOKfEFcR3vlYzZCaTPVwJsdr-O?usp=sharing)
    > Notebok solo disponible para FIUBENSES (por ahora)
    """
)

st.markdown("## Conclusiones")

st.markdown(
    """
    **QR** es un modelo de por sí ya eficiente y bastante útil, pero tiene la desventaja de que no da garantías de coberturas marginal teórica, también tiene otra ventaja y es que sus cuantiles intentan tener cobertura condicional.

    Bien, ¿pero qué ventajas supone la garantía de cobertura condicional? Básicamente me da intervalos más eficientes, para ello muchas veces se debe suponer una distribución y esto atenta con CP.

    Por ello CP no tiene garantía de cobertura condicional, aunque sí la garantía marginal, esto hace que los intervalos no sean eficientes y no se adapten a los datos (pero se puede jugar con esto).
    

    "Sin embargo, dado que la regresión cuantílica base intenta aprender los cuantiles condicionales verdaderos, la versión conformalizada (**CQR**) 
    hereda este buen comportamiento, proporcionando intervalos que tienen una cobertura "casi condicional" 
    de forma asintótica, pero manteniendo siempre la garantía marginal exacta para muestras finitas"
    """
)


st.markdown("## Bonus: Conformal Prediction para Forecasting")

st.markdown(
    """
    La idea de CP se puede extender a problemas de forecasting, es decir, predecir valores futuros de una serie temporal.

    En este caso, es simplemente aplicar lo visto en el caso "airline-passengers" el cual predice valores futuros de una serie temporal.

    Para armar los sets de datos, 
        - Se toma una ventana de 12 observaciones pasadas para predecir el siguiente valor.
        - Se asume estacionariedad en los datos.
        - Se asume que ergodicidad.
        - En este caso no se puede asumir intercambiabilidad, el orden si importa.

    Luego se aplica Conformal Quantile Regression (CQR) para obtener intervalos de predicción.
    
    """ 
)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(PATH_IMAGES + "QR_forecasting.png", caption="Regresión cuantílica", width="stretch")

with col2:
    st.image(PATH_IMAGES + "CQR_forecasting.png", caption="Conformal Quantile Regression", width="stretch")

with col3:
    st.image(PATH_IMAGES + "coverage_forecasting.png", caption="Cobertura empírica", width="stretch")

st.markdown("""
> Nota: Este metodo puede que sea Autorregresión Cuantílica (QAR)
> Notebook: [cqr_forecasting.ipynb](https://colab.research.google.com/drive/11mivutLu5lC8ovXXgRBmQPQ-T9F7VIC1?usp=sharing)
> Notebok solo disponible para FIUBENSES (por ahora)
""")

st.markdown("""
**Referencias:**
- [A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification](https://arxiv.org/pdf/2107.07511)
- [Conformal Quantile Regression](https://arxiv.org/pdf/1905.03222)
- [Tutorial on CP](https://claireboyer.github.io/tutorial-conformal-prediction/slides_cp.pdf)
- [A gentle introduction of conformal time series forecasting](https://arxiv.org/pdf/2511.13608)
""")
