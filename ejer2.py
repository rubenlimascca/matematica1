import numpy as np
############### 2  A)
# A. Generar los Nodos y los Valores de la Función
# Genera los nodos x = [-2.0, -1.5, ..., 1.5, 2.0] (9 puntos)
x_nodes = np.arange(-2.0, 2.1, 0.5) 

# Valores de la función y_i = cosh(x_i). Operación vectorizada.
y_nodes = np.cosh(x_nodes)

# B. Construir y Resolver el Sistema Lineal
# V: Matriz de Vandermonde de 9x9 (para polinomio de grado 8)
V_cosh = np.vander(x_nodes, increasing=True)

# a_cosh: Vector de 9 coeficientes [a0, a1, ..., a8]
# Resuelve V * a_cosh = y_nodes de forma vectorizada.
a_cosh = np.linalg.solve(V_cosh, y_nodes)

# Se invierte el orden para usar np.polyval (requiere [a8, a7, ..., a0])
a_cosh_rev = a_cosh[::-1] 

print("--- 2. Aproximación de cosh(x) ---")
print("\nNodos de Interpolación (xi):")
print(x_nodes)
print("\nValores de la Función (yi = cosh(xi)):")
print(y_nodes.round(4))
print("\nCoeficientes del Polinomio P(x) (de grado 8):")
print(a_cosh_rev.round(6))


############### 2  B)

import matplotlib.pyplot as plt

# Generar un rango denso de 200 puntos para la curva suave
x_plot = np.linspace(-2.5, 2.5, 200)

# Evaluar la función real f(x) = cosh(x)
y_real = np.cosh(x_plot)

# Evaluar el polinomio interpolante P(x)
# Se usa np.polyval con los coeficientes en orden decreciente
y_poly = np.polyval(a_cosh_rev, x_plot)

# Graficar
plt.figure(figsize=(9, 6))
plt.plot(x_plot, y_real, label='$f(x) = \cosh(x)$ (Función Real)', color='green', linestyle='--')
plt.plot(x_plot, y_poly, label='Polinomio Interpolante $P(x)$ (Grado 8)', color='red')
plt.plot(x_nodes, y_nodes, 'ko', markersize=4, label='Nodos de Interpolación (9 puntos)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproximación de $\cosh(x)$ mediante Interpolación Polinómica')
plt.grid(True)
plt.legend()
plt.ylim(0, 5.5) # Limitar el eje Y para una mejor visualización
plt.show()