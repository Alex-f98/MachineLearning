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
    el sobreajuste en modelos de machine learning.
    
    Vamos a ver un ejemplo de Ridge:
""")
#https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
st.latex(r"""
\min_{w} \underbrace{|\overbrace{Xw}^{\hat{y}} - y|_2^2}_{\text{Error de ajuste}} 
 + 
\underbrace{\alpha |w|_2^2}_{\substack{\text{Termino de}\\\text{regularización}}}
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

alpha  = st.slider("Alpha (regularización)", 0, 200, 20)
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

####

st.markdown("### Visualización de Regularización (L1 vs L2)")

# ecuaciones
st.markdown("El problema de optimización con regularización se puede expresar como:")
st.latex(r"""
\min_w J(w) + \lambda ||w||_q
""")
st.markdown("Donde:")
st.markdown("""
- $J(w)$: función de costo (por ejemplo, error cuadrático)
- $\lambda$: parámetro de regularización (controla el peso de la regularización)
- $||w||_q$: norma $L_q$ del vector de parámetros $w$

 La intuición matemática es clara, se intenta minimizar $J(w)^* = J(w) + \lambda ||w||_q$ osea que
 se tienen que minimizar ambos términos, por lo que minimizando solo el primer término puede llevar a valores de
 $w$ muy grandes que pueden causar overfitting (recordar: $\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n$), para evitar esto es que 
 se minimiza el segundo término que tenderá a minimizar los valores de $w$  manteniniendo así un trade-off entre ambos términos.
""")
st.markdown("Se presenta una función de costo convexa para la visualización")
st.latex(r"""
J(w) = (w - w_0)^T A (w - w_0)
""")

st.markdown("""
Bien, pero tambien hay una interpretación geometrica para entender esto.\\
En lugar de minimizar $\min_w J(w) + \lambda ||w||_q$, se puede pensar en minimizar $J(w)$ sujeto a una restricción de la norma de $w$.\\
Es decir, minimizar el error dentro de una región permitida del espacio de parametros.
""")


# sliders
lambda_reg = st.slider("λ regularización", 0.1, 5.0, 1.0)
angle      = st.slider("Rotación de la función de costo", 0.0, np.pi, 0.3)


# grid
x    = np.linspace(-10,10,400)
y    = np.linspace(-10,10,400)
X,Y  = np.meshgrid(x,y)

# centro de la función de error
cx, cy = 3, 3

# rotación
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])

XY = np.stack([X-cx, Y-cy], axis=0).reshape(2,-1)
rot = R @ XY
Z = (rot[0]**2 + 2*rot[1]**2).reshape(X.shape)

# Crear dos subplots lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

# Más curvas cerca del mínimo
levels = np.logspace(np.log10(Z.min()+0.1), np.log10(Z.max()), 15)

# Gráfico L1 (Lasso)
ax1.contour(X,Y,Z, levels=levels)
t = lambda_reg
diamond = np.array([
    [t,0],
    [0,t],
    [-t,0],
    [0,-t],
    [t,0]
])
ax1.fill(diamond[:,0], diamond[:,1], color='lightblue', alpha=0.7)
ax1.plot(diamond[:,0], diamond[:,1], 'b', linewidth=2)
ax1.axhline(0, color='gray', alpha=0.3)
ax1.axvline(0, color='gray', alpha=0.3)
ax1.set_xlim(-10,10)
ax1.set_ylim(-10,10)
ax1.set_xlabel("$w_1$")
ax1.set_ylabel("$w_2$")
ax1.tick_params(axis='y', labelsize=9)
ax1.set_title("L1 (Lasso)")

# Gráfico L2 (Ridge)
ax2.contour(X,Y,Z, levels=levels)
theta = np.linspace(0, 2*np.pi, 200)
r = lambda_reg
ax2.fill(r*np.cos(theta), r*np.sin(theta), color='lightblue', alpha=0.7)
ax2.plot(r*np.cos(theta), r*np.sin(theta), 'b', linewidth=2)
ax2.axhline(0, color='gray', alpha=0.3)
ax2.axvline(0, color='gray', alpha=0.3)
ax2.set_xlim(-10,10)
ax2.set_ylim(-10,10)
ax2.set_xlabel("$w_1$")
ax2.set_yticks([])
ax2.set_title("L2 (Ridge)")

plt.tight_layout()
st.pyplot(fig)

st.info("Referencia: [Bishop - 9.2.2 Generalized weight decay](https://www.bishopbook.com/)")