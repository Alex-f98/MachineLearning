import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from .numeric_solve import gradiente_descendiente


class Himmelblau:
    def __init__(self, x=np.linspace(-6, 6, 400), y=np.linspace(-6, 6, 400)):
      """
      Funcion Himmelblau
      f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
      Params:
        x: array de valores x
        y: array de valores y
      """
      self.x = x
      self.y = y

      self.X, self.Y = np.meshgrid(self.x, self.y) 
      self.Z = self.himmelblau(self.X, self.Y)
      
    def himmelblau(self, x, y):
      return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    def grad_himmelblau(self, x, y):
      df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
      df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
      return np.array([df_dx, df_dy])

    def df(self, xy):
      return self.grad_himmelblau(xy[0], xy[1])

    def plot_2d_himmelblau_with_trajectory(self, start_x, start_y, lr=0.01):
      # niveles de contorno
      contour_levels = np.concatenate(( np.arange(0, 10), np.array([15, 20, 25, 30]), np.arange(35, 500, 50) )) 
      # meshgrid 
      start = (start_x, start_y) 
      trajectory, _ = gradiente_descendiente(start, self.df, lr=lr, max_iter=50) 
      # grafico
      fig, ax = plt.subplots(figsize=(8, 6)) 
      ax.contour(self.X, self.Y, self.Z, levels=contour_levels, cmap='viridis') 
      ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-') 
      ax.set_title(f'Gradiente Descendente con Learning Rate = {lr}') 
      ax.set_xlabel('X1') 
      ax.set_ylabel('X2') 
      ax.set_xlim(-6, 6) 
      ax.set_ylim(-6, 6) 
      return fig

    def plot_3d_himmeblau(self):
      # Gráfico 3D con Plotly
      fig = go.Figure(data=[go.Surface(z=self.Z, x=self.X, y=self.Y, showscale=False)])
      fig.update_layout(
          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
          width=800, height=800
      )
      return fig

    def plot_2d_levels_himmeblau(self):
      contour_levels = np.concatenate((np.arange(0, 10), np.array([15,20,25,30]), np.arange(35, 500, 50)))
      fig, ax = plt.subplots(figsize=(8, 6))
      countour = ax.contour(self.X, self.Y, self.Z, levels=contour_levels, cmap='viridis')
      ax.set_title('Curvas de nivel función Himmelblau')
      ax.set_xlabel('X1')
      ax.set_ylabel('X2')
      ax.set_xlim(-6, 6)
      ax.set_ylim(-6, 6)
      return fig


class Polinomio1D():
  def __init__(self, x=np.linspace(-6, 6, 400), max_iter=100, convex=False):
    self.polinomio = self.polinomio_no_convexo if not convex else self.polinomio_convexo
    self.convex = convex
    self.x = x
    self.y = self.polinomio(x)
    self.max_iter = max_iter
    self._cache = {}

  def polinomio_no_convexo(self, x):
    """ f(w) = w⁴ - 6w² + 3w """
    return x**4 - 6*x**2 + 3*x

  def polinomio_convexo(self, x):
    """ f(w) = w² """
    return 2*x**2 -14

  def df(self, x):
    """ df(w) = 4w³ - 12w + 3 """
    return 4*x**3 - 12*x + 3
  
  def solve(self, start_w, lr=0.01):
    return gradiente_descendiente(start_w, self.df, lr=lr, max_iter=self.max_iter)

  def plot_2d_polinomio_with_trajectory(self, start_w, lr=0.01):
    """ Genero grafico 2D de f(w) vs w con la trayectoria del gradiente descendente para un lr dado"""

    trajectory, w_final = self.solve(start_w, lr=lr) 
    # grafico
    fig, ax = plt.subplots(figsize=(8, 8)) 
    ax.plot(self.x, self.y, lw=3)
    ax.plot(trajectory, self.polinomio(trajectory), 'ro-', label=f'w_final     = {w_final:.2f} \n steps      = {len(trajectory)-1} \n max_iter = {self.max_iter}') 
    ax.set_title(f'Gradiente Descendente con Learning Rate = {lr}') 
    ax.set_xlabel('X1') 
    ax.set_ylabel('X2') 
    ax.set_xlim(-4, 4) 
    ax.set_ylim(-15, 15) 
    ax.legend()
    return trajectory, w_final, fig

  def _solve_polynomial(self, w_0, lr):
    """Cache de la solucion, emula el @st.cache_data pero internamente"""
    # Crear clave única para el cache: (tuple_w_0, lr)
    cache_key = (tuple(w_0) if isinstance(w_0, (list, np.ndarray)) else (w_0,), lr)
    
    if cache_key not in self._cache:
        self._cache[cache_key] = self.solve(w_0, lr)
    
    return self._cache[cache_key]

  def dinamic_plot_2d_polinomio_with_trajectory(self, start_w, it, lr=0.01):
    trayectory_w, _ = self._solve_polynomial(start_w, lr)
    sns.set_theme(style="white", context="talk")  # context: paper, notebook, talk, poster

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(self.x, self.y, lw=7, color=sns.color_palette("dark")[0], label="Polinomio")

    # Trayectoria del gradiente
    ax.plot(
        trayectory_w[:it],
        self.polinomio(trayectory_w[:it]),
        's-',
        lw=7,
        color=sns.color_palette("dark")[1],
        label=f'w_final = {trayectory_w[:it][-1]:.3f}\nsteps = {it}\nmax_steps = {self.max_iter}'
    )
    ax.set_xlabel(rf'$\theta$', fontsize=14)
    ax.set_ylabel(rf'$J(\theta)$', fontsize=14)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-15, 15)
    ax.legend(fontsize=17, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return fig

    
