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

st.markdown(r"""
    La regularización Ridge añade una penalización L2 a la función de costo:
    
    $$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$$
    
    Esto crea una región de restricción circular en el espacio de parámetros.
    Mientras mayor sea $\alpha$, más pequeña será la región permitida.
""")

# Crear la visualización 2D
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Curvas de nivel de la penalización Ridge
theta1 = np.linspace(-3, 3, 100)
theta2 = np.linspace(-3, 3, 100)
THETA1, THETA2 = np.meshgrid(theta1, theta2)

# Función de penalización Ridge: alpha * (theta1^2 + theta2^2)
ridge_penalty = alpha * (THETA1**2 + THETA2**2)

# Dibujar curvas de nivel
contour = ax1.contour(THETA1, THETA2, ridge_penalty, levels=10, colors='blue', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)
ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.set_title(f'Curvas de nivel de penalización Ridge\n(α = {alpha})')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Marcar el origen
ax1.plot(0, 0, 'ro', markersize=8, label='Origen (óptimo regularizado)')
ax1.legend()

# Subplot 2: Región de restricción y ejemplo de optimización
# Dibujar la región de restricción (círculo)
circle = plt.Circle((0, 0), np.sqrt(1/alpha) if alpha > 0 else 10, 
                    fill=False, edgecolor='red', linewidth=2, 
                    label=f'Restricción Ridge: ||θ||² ≤ {1/alpha:.3f}' if alpha > 0 else 'Sin restricción')
ax2.add_patch(circle)

# Simular trayectoria de optimización hacia un mínimo
theta_optimal = np.array([1.5, 1.2])  # mínimo sin regularización
theta_ridge = theta_optimal * (1 / (1 + alpha * 0.1))  # aproximación del efecto ridge

# Dibujar trayectoria
ax2.arrow(0, 0, theta_optimal[0], theta_optimal[1], 
          head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
          alpha=0.5, label='Mínimo sin regularización')
ax2.arrow(0, 0, theta_ridge[0], theta_ridge[1], 
          head_width=0.1, head_length=0.1, fc='red', ec='red', 
          linewidth=2, label='Mínimo con Ridge')

ax2.set_xlabel(r'$\theta_1$')
ax2.set_ylabel(r'$\theta_2$')
ax2.set_title('Efecto de la regularización en el espacio de parámetros')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend()
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)

st.pyplot(fig2)

st.markdown("""
**Interpretación:**
- **Izquierda:** Las curvas de nivel muestran la penalización Ridge. Valores más alejados del origen tienen mayor penalización.
- **Derecha:** La restricción Ridge limita los coeficientes a una región circular. El óptimo regularizado (rojo) está más cerca del origen que el óptimo sin regularización (azul).
- **Efecto del α:** Mayor α → región más pequeña → coeficientes más pequeños → mayor regularización.
""")