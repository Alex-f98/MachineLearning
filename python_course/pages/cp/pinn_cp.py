import streamlit as st

PATH_IMAGES = "python_course/image/cp_img/"

st.markdown("""
Índice:

* [Introducción](#introduccion)
* [Fundamentos Teóricos](#fundamentos-teoricos)
  * [Ecuación de Poisson 1D](#ecuacion-de-poisson-1d)
  * [Physics-Informed Neural Networks](#physics-informed-neural-networks)
* [Incertidumbre Cuantificable](#incertidumbre-cuantificable)
  * [Inferencia Bayesiana Variacional](#inferencia-bayesiana-variacional)
  * [Limitaciones del Enfoque Bayesiano](#limitaciones-del-enfoque-bayesiano)
* [Predicción Conformal en PINNs](#prediccion-conformal-en-pinns)
  * [Calibración de Intervalos](#calibracion-de-intervalos)
  * [Heurísticas de No-Conformidad](#heuristicas-de-no-conformidad)
* [Resultados y Aplicaciones](#resultados-y-aplicaciones)
* [Conclusiones](#conclusiones)
---
""")



st.markdown("## Introducción - PINN")

st.markdown(
    """
    Una PINN (Physics-Informed Neural Network) es una red neuronal diseñada para resolver 
    ecuaciones diferenciales parciales (PDE) integrando leyes físicas directamente en su proceso de entrenamiento

    A diferencia de las redes neuronales convencionales que solo aprenden de datos, 
    las PINNs actúan como un solucionador libre de mallas (mesh-free) que utiliza la información de las leyes físicas 
    para guiar el aprendizaje incluso en regiones donde no hay datos observados.

    La representación más general de una **Red Neuronal Informada por la Física (PINN)** no se limita a una sola ecuación, 
    sino a un marco de trabajo que combina un problema de valor inicial y de contorno con una **función de pérdida multiobjetivo**.

    ### 1. El problema físico gobernante
    Una PINN busca aproximar una solución $u$ para un sistema definido por una **ecuación diferencial parcial (PDE)** genérica:

    *   **Ecuación diferencial:** $\mathcal{L}[u](x, t) = f(x, t)$ para $(x, t) \in \Omega \\times (0, T]$, donde $\mathcal{L}$ es un operador diferencial (posiblemente no lineal) y $f$ es un término fuente conocido.
    *   **Condición inicial:** $u(x, 0) = u_0(x)$ para $x \in \Omega$.
    *   **Condiciones de contorno:** $\mathcal{B}[u](x, t) = g(x, t)$ para $(x, t) \in \partial\Omega \\times (0, T]$, donde $\mathcal{B}$ es un operador de contorno.

    ### 2. La ecuación de representación (Función de Pérdida)
    La "ecuación" que representa el entrenamiento y la esencia de la PINN es su **función de pérdida empírica total**, la cual se minimiza para encontrar 
    los parámetros óptimos $\\theta$ de la red neuronal $u_\\theta$:

    $$\mathbf{Loss(\\theta) = \lambda_{data} L_{data}(\\theta) + \lambda_{pde} L_{pde}(\\theta) + \lambda_{ic} L_{ic}(\\theta) + \lambda_{b} L_{bc}(\\theta)}$$

    Donde los componentes se definen de la siguiente manera:

    *   **$L_{data}(\\theta)$:** Es la pérdida de fidelidad a los datos, que mide la diferencia entre la salida de la red y las observaciones reales o medidas de sensores.
    *   **$L_{pde}(\\theta)$:** Es la **pérdida de la física**, que evalúa cuánto se desvía la red neuronal de la ecuación diferencial en puntos de colocación dentro del dominio. Su objetivo es minimizar el residuo $\mathcal{R}(u_\theta) = \mathcal{L}[u_\theta] - f$.
    *   **$L_{ic}(\\theta)$ y $L_{bc}(\\theta)$:** Son las pérdidas asociadas al cumplimiento de las condiciones iniciales y de contorno, respectivamente.
    *   **$\lambda$ (pesos):** Son coeficientes no negativos que equilibran la importancia de cada término durante la optimización.

    > Mas información sobre PINNs en el siguiente paper: [Solving the wave equation with physics-informed deep learning](https://arxiv.org/pdf/2006.11894)
    """
)

st.markdown("## Ecuación de Poisson 1D")

st.markdown(
    """
    A continuación se quiere resolver el problema de Posisson para una dimensión.
    
    La ecuación de Poisson es una EDP elíptica fundamental que aparece en numerosos problemas físicos:
    
    $$
    u''(x) = f(x)
    $$
    
    donde:
    - $u(x)$ es la función desconocida que queremos determinar
    - $f(x)$ es el término fuente conocido
    - $u''(x)$ representa la segunda derivada de $u$ respecto a $x$
    
    
    Para este problema, imponemos condiciones de frontera tipo Dirichlet:
    
    $$
    u(0) = u_0, \\quad u(1) = u_1
    $$
    
    
    En este caso, para validación, utilizamos una solución analítica conocida:
    
    $$
    u_{verdad}(x) = \\sin(\\pi x)
    $$
    
    en donde el término fuente es:
    
    $$
    f(x) = -\\pi^2 \\sin(\\pi x)
    $$
    
    La elección de $\\sin(\\pi x)$ es estratégica porque cumple automáticamente las condiciones de frontera y es infinitamente diferenciable.
    """
)

st.markdown("### Physics-Informed Neural Networks (PINNs) para Poisson 1D")
st.image(PATH_IMAGES + "PINN_esquema.png", caption="Esquema de PINN ", width="stretch")
st.markdown(
    """
    Para resolver el problema de Poisson 1D con PINNs, se utiliza una red neuronal para aproximar la solución $u(x)$.

    Se debe definir las funciones de perdida a minimizar:
    
     - **Pérdida de Datos**
        $$
        \\mathcal{L}_{data} = \\sum_{i} \\frac{(y_i - \\hat{y}_i)^2}{2\\sigma^2} + \\frac{1}{2}\\log(2\\pi\\sigma^2)
        $$
    
        Mide el ajuste del modelo a los datos observados.
    
    - **Pérdida Física**
        $$
        \\mathcal{L}_{pde} = \\mathbb{E}_{q(\\theta)}\\left[\\frac{1}{N_c}\\sum_{j=1}^{N_c} (u''(x_j) - f(x_j))^2\\right]
        $$

        Impone la satisfacción de la ecuación diferencial en puntos de colocación.
    
    - **Pérdida de Frontera**
        $$
        \\mathcal{L}_{bc} = (u(0)-u_0)^2 + (u(1)-u_1)^2
        $$
        
        Garantiza el cumplimiento de las condiciones de contorno.

    - **Pérdida ELBO (Evidence Lower Bound)**
        $$
        \\mathcal{L}_{elbo} = -\\mathbb{E}_{q_\\phi(\\theta)} [ \\log p(\\mathcal{D}_{data}|\\theta) ] + \\text{KL} ( q_\\phi(\\theta) || p_0(\\theta) )
        $$
    
        Combina verosimilitud y regularización para incertidumbre.
    
    #### Función de Pérdida Total
    $$
    \\mathcal{L} = \\lambda_{pde} \\cdot \\mathcal{L}_{pde} + \\lambda_{bc} \\cdot \\mathcal{L}_{bc} + \\lambda_{data} \\cdot \\mathcal{L}_{data} + \\lambda_{elbo} \\cdot \\mathcal{L}_{elbo}
    $$
    """
)

st.markdown("## Incertidumbre Cuantificable")

st.markdown(
    """
    **Proceso de Entrenamiento (Inferencia Variacional)**
    
    El objetivo central del entrenamiento no es encontrar un único valor óptimo para los pesos $\\theta$, sino aprender una **distribución de probabilidad** sobre ellos.
    para ello se define una probabilidad a priori $p_0(\\theta)$ y se quiere encontrar la posteriori $p(\\theta|\\mathcal{D}) = \\frac{p(\\mathcal{D}|\\theta)p_0(\\theta)}{p(\\mathcal{D})}$.

    > Notar que $p(\\mathcal{D}) = \\int p(\\mathcal{D}|\\theta)p_0(\\theta)d\\theta$ es intratable, por lo que se utiliza una aproximación $q_\\phi(\\theta)$.
    
    #### Definición del Posterior Aproximado ($q_\\phi(\\theta)$)
    
    Se asume una familia de distribuciones gaussianas totalmente factorizadas donde cada peso $\\theta_j$ tiene su propia media $\\mu_j$ y desviación estándar $\\sigma_j$. Para garantizar que $\\sigma_j$ sea siempre positiva, se utiliza la transformación **softplus**:
    
    $$
    \\sigma_j = \\log(1 + \\exp(\\rho_j))
    $$
    
    #### Optimización del ELBO
    
    El entrenamiento consiste en minimizar el **L-ELBO Negativo** mediante el optimizador **Adam**. Este proceso equilibra dos fuerzas:

    $$
    \\mathcal{L}_{neg} = -\\mathbb{E}_{q_\\phi(\\theta)} [ \\log p(\\mathcal{D}_{data}|\\theta) ] + \\text{KL} ( q_\\phi(\\theta) || p_0(\\theta) )
    $$

    1. **Fidelidad de los datos (Expected Log-likelihood):** Se asegura de que, en promedio, los parámetros muestreados de $q_\\phi$ expliquen bien tanto los datos de entrenamiento observados como el residuo de la ecuación diferencial (la física).
    
    2. **Penalización de complejidad (Divergencia KL):** Actúa como un regularizador, impidiendo que la distribución aprendida se aleje demasiado de una distribución previa simple (usualmente una normal estándar), lo que evita el sobreajuste.
    
    #### Truco de Reparametrización
    
    Para poder aplicar retropropagación (backpropagation) a través de un proceso de muestreo estocástico, se utiliza la fórmula:
    
    $$
    \\theta = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)
    $$
    
    Esto permite que el gradiente fluya directamente hacia los parámetros variacionales $\\mu$ y $\\rho$ de manera determinista y con baja varianza.
    
    ### Proceso de Inferencia (Muestreo Predictivo)
    
    Una vez que el modelo ha convergido y los parámetros $\\phi = \\{\\mu, \\rho\\}$ están optimizados, el modelo no produce una única solución, sino una **distribución predictiva** para cada punto $x$ en el dominio.
    
    #### Muestreo de Monte Carlo
    
    Para realizar una predicción en un nuevo punto $x_{new}$, se extraen **$M$ muestras** de los pesos de la red a partir de la distribución optimizada:
    
    $$
    \\theta^{(m)} \\sim q_\\phi(\\theta) \\quad \\text{para} \\quad m = 1, \\dots, M
    $$
    
    #### Cálculo de la Media Predictiva
    
    Se ejecutan $M$ pases hacia adelante (forward passes) a través de la red con cada conjunto de pesos muestreado. La predicción final es el promedio de estas salidas:
    
    $$
    \\mu_{BAY}(x_{new}) = \\frac{1}{M} \\sum_{m=1}^{M} f_{\\theta^{(m)}}(x_{new})
    $$
    
    #### Cálculo de la Varianza Predictiva (Incertidumbre)
    
    La incertidumbre epistémica (falta de conocimiento del modelo) se cuantifica mediante la varianza de las $M$ predicciones:
    
    $$
    \\sigma^2_{BAY}(x_{new}) = \\frac{1}{M} \\sum_{m=1}^{M} \\|f_{\\theta^{(m)}}(x_{new}) - \\mu_{BAY}(x_{new})\\|_2^2
    $$
    
    ### Conexión con la Predicción Conforme (CP)
    
    Es crucial entender que la varianza calculada en la inferencia bayesiana ($\\sigma^2_{BAY}$) suele ser **demasiado optimista** (subestima el error real) en problemas con pocos datos.
    
    En el caso de Poisson 1D, esta varianza bayesiana se toma como una **puntuación de incertidumbre heurística**. El algoritmo de Predicción Conforme toma esta varianza y la utiliza para calcular los "scores" en el conjunto de **calibración** (30 muestras que no se usaron en el entrenamiento). Al encontrar el cuantil adecuado de estos errores corregidos por la varianza bayesiana, CP genera intervalos de confianza que sí garantizan una cobertura del 95%, corrigiendo la sobreconfianza inicial del modelo bayesiano.
    """
)

st.markdown(
    """
    ### Limitaciones del Enfoque Bayesiano
    
    A pesar de su elegancia teórica, las PINNs bayesianas presentan limitaciones prácticas significativas:
    
    #### Fuentes de Subestimación de Incertidumbre
    
    1. **Aproximación Media-Campo**: La independencia entre pesos es restrictiva y subestima la covarianza
    
    2. **Familia Variacional**: Distribución gaussiana puede ser inadecuada para la posterior verdadera
    
    3. **Optimización Local**: Mínimos locales en ELBO pueden atrapar al optimizador
    
    4. **Modelo de Ruido Simplificado**: **Homocedasticidad asumida** puede violar la realidad
    
    #### Consecuencias Prácticas
    
    - **Cobertura Real**: Generalmente 50-60% vs 95% nominal
    - **Intervalos Optimistas**: Demasiado estrechos para el nivel de confianza especificado
    - **Sobreconfianza Sistemática**: El modelo es demasiado seguro de sus predicciones

    """
)

st.markdown("""## Métricas de Evaluación""")

st.markdown("""

    **Cobertura Empírica:**
    $$
    \\text{Cobertura} = \\frac{1}{N}\\sum_{i=1}^N \\mathbb{I}[y_i \\in [\\hat{L}(x_i), \\hat{U}(x_i)]]
    $$
    
    **Precisión (Sharpness):**
    $$
    \\text{Precisión} = \\frac{1}{N}\\sum_{i=1}^N (\\hat{U}(x_i) - \\hat{L}(x_i))
    $$
    
    **Interval Score:**
    $$
    \\text{IS} = \\frac{1}{N}\\sum_{i=1}^N [\\hat{U}(x_i) - \\hat{L}(x_i)] + \\frac{2}{\\alpha}\\sum_{i=1}^N \\max(0, \\hat{L}(x_i)-y_i, y_i-\\hat{U}(x_i))
    $$
    """
)

st.markdown("## Predicción Conformal en PINNs")

st.markdown(
    """
    ### Calibración de Intervalos
    
    La Predicción Conformal (CP) es un framework que proporciona **garantías de cobertura válidas** bajo condiciones de intercambiabilidad:
    
    $$
    \\mathbb{P}[Y_{n+1} \\in \\hat{C}_{\\alpha}(X_{n+1})] \\geq 1 - \\alpha
    $$
    
    #### Ventajas Teóricas Fundamentales
    
    1. **Garantías Exactas**: Cobertura finita válida (no asintótica)
    2. **Sin Supuestos Distribucionales**: No requiere normalidad de errores
    3. **Adaptabilidad**: Se ajusta a la complejidad local del problema
    4. **Compatibilidad**: Funciona con cualquier modelo base (black-box)

    > Nota: esto fue mejor explicado en la sección "intro cp"

    ### Algoritmo de Calibración Conformal
    
    -  **Paso 1: Cálculo de Puntuaciones de Conformidad**
    
        Se utilizan heurísticas para medir la no-conformidad en el conjunto de calibración:
        
        $$
        \\text{score}_i = \\frac{|y_i - \\hat{y}_i|}{\\text{width}_i}
        $$
        
        donde:
        - $|y_i - \\hat{y}_i|$: Residuo absoluto en punto de calibración
        - $\\text{width}_i$: Ancho del intervalo de predicción bayesiano
    
    - **Paso 2: Cuantil de Calibración**
    
        $$
        q_{\\alpha} = \\left\\lceil\\frac{(n_{cal}+1)(1-\\alpha)}{n_{cal}}\\right\\rceil / n_{cal}
        $$
    
    - **Paso 3: Construcción de Intervalos Calibrados**
    
        $$
        \\hat{C}_{\\alpha}(x) = \\hat{y}(x) \\pm q_{\\alpha} \\cdot u(x)
        $$
        
        donde $u(x)$ es la heurística de incertidumbre del modelo base.
        """
)


st.markdown("### Heurísticas de No-Conformidad")
st.markdown(
    """
    La nocion de heuristica de no-conformidad es fundamental en el proceso de calibración y realmente tiene pocas restricciones.

    Sin embargo juega un papel clave en la eficiencia de los intervalos generados.

    
    - 1. **Heurística 'raw_std' (Utilizada en Demo)**
    
        **Concepto**: Utilizar la incertidumbre nativa del modelo bayesiano
        
        **Métrica**:
        $$
        \\text{score}_i = \\frac{|y_i - \\hat{y}_i|}{\\text{width}_i}
        $$
        
        **Fundamento**: Normaliza el error por la incertidumbre estimada, penalizando predicciones sobreconfidentes.
        
    - 2. **Heurística 'feature' (k-NN en Espacio de Entrada)**
    
        **Métrica**:
        $$
        \\text{score}_i = \\frac{|y_i - \\hat{y}_i|}{\\text{dist}_k(X_i, X_{train})}
        $$
        
        **Interpretación**: Puntos en regiones "similares" a datos de entrenamiento deberían tener menor incertidumbre.
        
    - 3. **Heurística 'latent' (k-NN en Espacio Latente)**
        
        **Métrica**:
        $$
        \\text{score}_i = \\frac{|y_i - \\hat{y}_i|}{\\text{dist}_k(H(X_i), H(X_{train}))}
        $$
        
        donde $H(X)$ es la representación en la última capa oculta de la red neuronal.
    
    ### Propiedades Teóricas del Resultado
    
    #### Garantías de Cobertura
    
    1. **Cobertura Exacta (Bajo Intercambiabilidad)**:
       $$
       \\mathbb{P}[Y_{n+1} \\in \\hat{C}_{\\alpha}(X_{n+1})] \\geq 1 - \\alpha
       $$
    
    2. **Conservatismo Controlado**: El factor $q_{\\alpha}$ ajusta exactamente la cobertura
    
    3. **Adaptabilidad Local**: Intervalos más anchos donde el modelo es menos confiable
    
    #### Eficiencia y Precisión
    
    - **Preserva Ranking**: CP no altera el ordenamiento relativo de incertidumbres
    - **Ajuste Multiplicativo**: Factor escala uniformemente todos los intervalos
    - **Mantenimiento de Forma**: Se mantiene la estructura del modelo base
    """
)

st.markdown("## Resultados y Aplicaciones")

st.image(PATH_IMAGES + "vi_pinn_cp_pred.png", caption="Predicción con PINN Calibrado", width="stretch")

st.markdown(
    """
    ### Efecto de la Calibración Conformal
        
    - **Métricas Agregadas (α=0.05):**

        | Métrica | Sin CP | Con CP | Mejora |
        |---------|--------|--------|--------|
        | Coverage | 53.3% | 97.3% | +44.0% |
        | Sharpness | 0.216 | 0.571 | -0.355 |
        | Interval Score | 1.564 | 0.654 | -58.2% |
        | ACD | 0.256 | 0.072 | -71.9% |



    - **Éxitos del CP:**
        1. **Calibración Perfecta:** Cobertura prácticamente nominal
        2. **Garantías Robustas:** Cumplimiento de requisitos teóricos
        3. **Eficiencia Razonable:** Precisión aceptable bajo restricciones
        4. **Consistencia:** Rendimiento estable a través de $.\\alpha$

    - **Costos del CP:**
        1. **Menor Precisión:** Intervalos más anchos (inevitable)
        2. **Complejidad Adicional:** Requiere conjunto de calibración
        3. **Ligero Conservatismo:** Cobertura ligeramente superior a nominal

    Esto tambien se puede ver en el siguiente grafico.

    Vemos que la covertura es ampliamente superadora de la cobertura nominal, lo que indica un comportamiento conservador a costa de
    que el sharpness (ancho del intervalo) sea mayor, lo cual disminuye la eficiencia del intervalo.

    """
)

st.image(PATH_IMAGES + "metrics_vi_pinn_cp1D.png", caption="Métricas de CP", width="stretch")

st.markdown(
    """
    Finalmente, se muestra la covertura para diferentes valores de alpha.    
    """
)
st.image(PATH_IMAGES + "coverage_vi_pinn_cp1D.png", caption="Cobertura para diferentes valores de alpha", width="stretch")

st.markdown(
    """
    Se puede observar que la cobertura es mayor que la cobertura nominal (curva azul) para todos los valores de alpha, 
    aunque comparados con lo covertura deseada (curva puenteada negra) este queda por debajo para niveles de confianza pequeños.
    """
)
