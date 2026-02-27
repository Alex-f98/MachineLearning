import numpy as np
import matplotlib.pyplot as plt
from .numeric_solve import gradiente_descendiente

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def grad_himmelblau(x, y):
      df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
      df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
      return np.array([df_dx, df_dy])

def df(xy):
  return grad_himmelblau(xy[0], xy[1])

# Función para graficar trayectoria 
def plot_himmelblau_with_trajectory(start_x, start_y, lr=0.01):
     x = np.linspace(-6, 6, 400) 
     y = np.linspace(-6, 6, 400) 
     X, Y = np.meshgrid(x, y) 
     Z = himmelblau(X, Y) 
     start = (start_x, start_y) 
     trajectory, _ = gradiente_descendiente(start, df, lr=lr, max_iter=50) 
     contour_levels = np.concatenate(( np.arange(0, 10), np.array([15, 20, 25, 30]), np.arange(35, 500, 50) )) 
     fig, ax = plt.subplots(figsize=(8, 6)) 
     ax.contour(X, Y, Z, levels=contour_levels, cmap='viridis') 
     ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-') 
     ax.set_title(f'Gradiente Descendente con Learning Rate = {lr}') 
     ax.set_xlabel('X1') 
     ax.set_ylabel('X2') 
     ax.set_xlim(-6, 6) 
     ax.set_ylim(-6, 6) 
     return fig
