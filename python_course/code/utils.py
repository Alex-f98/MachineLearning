import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from .numeric_solve import gradiente_descendiente


#=================================================================================#
#
#  Funciones para graficar gradientes.
#
#=================================================================================#

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

    
#=================================================================================#
#
#   Funciones para graficar normalizacion
#
#=================================================================================#

class NormalizationExperiments:
    """Clase para visualizar el efecto de la normalización en la función de costo ECM"""
    
    def __init__(self):
        pass
        self.w1_optimo = -np.inf
        self.w2_optimo = -np.inf
    
    def normalize_data(self, X):
        """Normaliza los datos"""
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        return (X - X_mean) / X_std, X_mean, X_std
    
    def create_parameter_matrix(self, w1_min=-30, w1_max=30, w2_min=-30, w2_max=30, n_points=20):
        """Crea matriz de parámetros vectorizada"""
        w1_range = np.linspace(w1_min, w1_max, n_points)
        w2_range = np.linspace(w2_min, w2_max, n_points)
        W1, W2 = np.meshgrid(w1_range, w2_range)
        return W1, W2, np.c_[W1.ravel(), W2.ravel()]
    
    def calculate_ecm_vectorized(self, X, y, W_matrix):
        """Calcula ECM para todas las combinaciones de parámetros vectorizadamente"""
        y_pred_all = W_matrix @ X.T
        ecm_values = np.mean((y - y_pred_all) ** 2, axis=1)
        return ecm_values
    
    def _get_surface_data(self, X, y, normalize=False, w1_min=-30, w1_max=30, w2_min=-30, w2_max=30, n_points=100):
        """Método interno para obtener datos de superficie (evita repetición)"""
        X_processed = X.copy()
        if normalize:
            X_processed, _, _ = self.normalize_data(X)
        
        W1, W2, W_matrix = self.create_parameter_matrix(w1_min, w1_max, w2_min, w2_max, n_points)
        ecm_values = self.calculate_ecm_vectorized(X_processed, y, W_matrix)
        J = ecm_values.reshape(W1.shape)
        return W1, W2, J
    
    def ecm_3d_vectorized(self, X, y, normalize=False, **kwargs):
        """Función principal vectorizada"""
        W1, W2, J = self._get_surface_data(X, y, normalize, **kwargs)
        
        title = f"ECM {'CON' if normalize else 'SIN'} Normalización"
        fig = go.Figure(data=[go.Surface(
          z=J, 
          x=W1, 
          y=W2, 
          colorscale='Blues',
          cmin=0,
          cmax=40000,
          colorbar=dict(
              title="ECM",
              thickness=20,
              len=0.7
          )
        )])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='w1',
                yaxis_title='w2',
                zaxis_title='ECM'
            ),
            width=600,
            height=400
        )
        return fig
    
    def plot_3d_subplot(self, X, y, w1_range, w2_range, J_range):
        """Grafica ambas superficies en subplot 1x2"""
        params = dict(w1_min=w1_range[0], w1_max=w1_range[1], 
                     w2_min=w2_range[0], w2_max=w2_range[1], n_points=50)
        
        fig_no_norm = self.ecm_3d_vectorized(X, y, normalize=False, **params)
        fig_norm = self.ecm_3d_vectorized(X, y, normalize=True, **params)
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=['ECM SIN Normalización', 'ECM CON Normalización']
        )
        
        fig.add_trace(fig_no_norm.data[0], row=1, col=1)
        fig.add_trace(fig_norm.data[0], row=1, col=2)
        
        fig.update_layout(
            title='Comparación: ECM SIN vs CON Normalización',
            scene=dict(
                xaxis_title='w1', yaxis_title='w2', zaxis_title='ECM',
                zaxis=dict(range=J_range)
            ),
            scene2=dict(
                xaxis_title='w1', yaxis_title='w2', zaxis_title='ECM',
                zaxis=dict(range=J_range)
            ),
            width=1200, height=500
        )
        return fig
    
    def plot_2d_comparison(self, X, y, w1_range=(-30, 30), w2_range=(-30, 30)):
        """Grafica contornos en 2D para comparar"""
        params = dict(w1_min=w1_range[0], w1_max=w1_range[1], 
                     w2_min=w2_range[0], w2_max=w2_range[1], n_points=50)
        
        W1_no, W2_no, J_no    = self._get_surface_data(X, y, normalize=False, **params)
        W1_yes, W2_yes, J_yes = self._get_surface_data(X, y, normalize=True, **params)

        w1_optimo_norm = self.w1_optimo * np.std(X[:, 0])
        w2_optimo_norm = self.w2_optimo * np.std(X[:, 1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        contour1 = ax1.contour(W1_no, W2_no, J_no, levels=20, cmap='viridis')
        ax1.scatter(self.w1_optimo, self.w2_optimo, color='red', s=100, label=f'({self.w1_optimo:.2f}, {self.w2_optimo:.2f})')
        ax1.hlines(self.w2_optimo, W1_no.min(), self.w1_optimo, colors='red', linestyles='dashed', linewidth=2)
        ax1.vlines(self.w1_optimo, W2_no.min(), self.w2_optimo, colors='red', linestyles='dashed', linewidth=2)
        ax1.clabel(contour1, inline=True, fontsize=8)
        ax1.set_title('ECM SIN Normalización')
        ax1.set_xlabel('w1'); ax1.set_ylabel('w2')
        ax1.grid(True, alpha=0.3)
        # Agregar anotación de texto en el punto óptimo
        ax1.annotate(f'({self.w1_optimo:.2f}, {self.w2_optimo:.2f})', 
                    xy=(self.w1_optimo, self.w2_optimo),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)
        
        contour2 = ax2.contour(W1_yes, W2_yes, J_yes, levels=20, cmap='viridis')
        ax2.hlines(w2_optimo_norm, W1_yes.min(), w1_optimo_norm, colors='red', linestyles='dashed', linewidth=2)
        ax2.vlines(w1_optimo_norm, W2_yes.min(), w2_optimo_norm, colors='red', linestyles='dashed', linewidth=2)
        ax2.scatter(w1_optimo_norm, w2_optimo_norm, color='green', s=100, label=f'({w1_optimo_norm:.2f}, {w2_optimo_norm:.2f})')
        ax2.clabel(contour2, inline=True, fontsize=8)
        ax2.set_title('ECM CON Normalización')
        ax2.set_xlabel('w1'); ax2.set_ylabel('w2')
        ax2.grid(True, alpha=0.3)
        # Agregar anotación de texto en el punto óptimo normalizado
        ax2.annotate(f'({w1_optimo_norm:.2f}, {w2_optimo_norm:.2f})', 
                    xy=(w1_optimo_norm, w2_optimo_norm),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_data_comparison(self, X, y):
        """Grafica datos originales vs normalizados en subplot"""
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        X_norm, mean, std = self.normalize_data(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SIN normalizar
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
        ax1.set_title('Datos SIN Normalización')
        ax1.set_xlabel('X₁'); ax1.set_ylabel('X₂')
        ax1.set_xlim(min_x[0] - 5, max_x[0] + 5)
        ax1.set_ylim(min_x[1] - 5, max_x[1] + 5)
        ax1.grid(True, alpha=0.3)
        
        # CON normalización
        scatter2 = ax2.scatter(X_norm[:, 0], X_norm[:, 1], alpha=0.6)
        ax2.set_title('Datos CON Normalización')
        ax2.set_xlabel('X₁ (normalizado)'); ax2.set_ylabel('X₂ (normalizado)')
        ax2.set_xlim(min_x[0] - 5, max_x[0] + 5)
        ax2.set_ylim(min_x[1] - 5, max_x[1] + 5)
        ax2.grid(True, alpha=0.3)
        
        # Agregar estadísticas
        ax1.text(0.02, 0.98, f'Media: [{np.mean(X[:, 0]):.2f}, {np.mean(X[:, 1]):.2f}]\nStd: [{np.std(X[:, 0]):.2f}, {np.std(X[:, 1]):.2f}]',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.text(0.02, 0.98, f'Media: [{np.mean(X_norm[:, 0]):.2f}, {np.mean(X_norm[:, 1]):.2f}]\nStd: [{np.std(X_norm[:, 0]):.2f}, {np.std(X_norm[:, 1]):.2f}]',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_data(self, w1_optimo, w2_optimo, n_samples=1000, scale1=2, scale2=20, mean1=5, mean2=5, noise_std=0.3):
        """Genera datos con parámetros específicos"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 2) * [scale1, scale2] + [mean1, mean2]
        y = w1_optimo * X[:,0] + w2_optimo * X[:,1] #+ np.random.randn(n_samples) * noise_std
        
        print(f"Parámetros óptimos: w1={w1_optimo}, w2={w2_optimo}")
        print(f"X - Media: {np.mean(X, axis=0)}, Std: {np.std(X, axis=0)}")

        self.w1_optimo = w1_optimo
        self.w2_optimo = w2_optimo
        
        return X, y

    def run_experiment(self, w1_optimo=-10, w2_optimo=-10, scale1=2, scale2=20):
        """Ejecuta experimento completo"""
        # Generar datos
        X, y = self.generate_data(w1_optimo, w2_optimo, scale1=scale1, scale2=scale2)
        
        # Definir rangos centrados en óptimos
        margin = 100
        w1_range = (w1_optimo - margin, w1_optimo + margin)
        w2_range = (w2_optimo - margin, w2_optimo + margin)
        J_range = [0, 40_000]
        
        # Gráficos
        fig_data = self.plot_data_comparison(X, y)
        fig_3d   = self.plot_3d_subplot(X, y, w1_range, w2_range, J_range)
        fig_2d   = self.plot_2d_comparison(X, y, w1_range, w2_range)
        
        return X, y, fig_data, fig_3d, fig_2d