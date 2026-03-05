from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("""
* **[Regularización Ridge](#regularizacion-ridge)**
""")

st.markdown("### Regularización Ridge")
st.markdown("""
    La regularización Ridge es una técnica de regularización que se utiliza para prevenir 
    el sobreajuste en modelos de regresión.
    
    Vamos a ver un ejemplo de Ridge:
""")

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

m  = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7*x1**2 - 2*x1 + 3 + np.random.randn(m,1)
#plt.scatter(x1,x2)
#plt.show()

def get_preds_ridge(x1, x2, alpha, degree=16):
  model = Pipeline([
      ('poly_feats',PolynomialFeatures(degree=degree)),
      ('ridge', Ridge(alpha=alpha))
  ])
  model.fit(x1,x2)
  return model.predict(x1)

# --- Visualización 1: Regresión con Ridge ---
st.markdown("#### Ajuste de regresión polinomial con regularización Ridge")

alpha = st.slider("Alpha (regularización)", 0, 200, 20)
degree = st.slider("Grado del polinomio", 1, 40, 16)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x1,x2,'b+', label='Datapoints')
preds = get_preds_ridge(x1,x2,alpha, degree)
 # Plot
ax.plot(sorted(x1[:,0]), preds[
      np.argsort(x1[:,0])
  ],'r-', lw=4, 
  label = 'Alpha   : {}\nDegree : {}'.format(alpha, degree))

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Regresión polinomial con regularización Ridge')
st.pyplot(fig)




# --- Visualización 2: Cueva de nivel 2D (Restricción Ridge) ---
st.markdown("#### Visualización de la restricción Ridge (Cueva de nivel 2D)")

st.markdown(
    r"""
    La regularización Ridge añade una penalización L2 a la función de costo:
    
    $$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$$
    
    Esto crea una región de restricción circular en el espacio de parámetros.
    Mientras mayor sea $\alpha$, más pequeña será la región permitida.
""")

st.markdown(
    """
    **Interpretación:**
    - **Izquierda:** Las curvas de nivel muestran la penalización Ridge. Valores más alejados del origen tienen mayor penalización.
    - **Derecha:** La restricción Ridge limita los coeficientes a una región circular. El óptimo regularizado (rojo) está más cerca del origen que el óptimo sin regularización (azul).
    - **Efecto del α:** Mayor α → región más pequeña → coeficientes más pequeños → mayor regularización.
    """
)
