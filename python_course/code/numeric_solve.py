import numpy as np

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
