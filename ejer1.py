import numpy as np
# ######  1   A)
# A. Definir los datos
x = np.array([-1, -2, 4, 2])
y = np.array([3, 3, 1, 4])

# B. Construir la Matriz de Vandermonde (V)
# np.vander(x, increasing=True) construye la matriz V
# con columnas [1, x, x^2, x^3] sin usar bucles explícitos.
V = np.vander(x, increasing=True)

# V es la matriz 4x4:
# [[ 1. -1.  1. -1.]
#  [ 1. -2.  4. -8.]
#  [ 1.  4. 16. 64.]
#  [ 1.  2.  4.  8.]]

# C. Resolver el Sistema Lineal V * a = y
# La función np.linalg.solve resuelve el sistema de ecuaciones lineales
# de forma vectorizada, encontrando el vector de coeficientes 'a'.
# a = [a0, a1, a2, a3]
a = np.linalg.solve(V, y)

print("--- 1. Polinomio Interpolante (Método Vandermonde) ---")
print("\nVector de Coeficientes a = [a0, a1, a2, a3]:")
# Redondeo para mejor visualización
print(a.round(6)) 

# El polinomio es: P(x) = a[0] + a[1]*x + a[2]*x^2 + a[3]*x^3
a0, a1, a2, a3 = a
print(f"\nPolinomio P(x) = {a0:.4f} + {a1:.4f}x + {a2:.4f}x^2 + {a3:.4f}x^3")


####################
# ######  1   B)
import matplotlib.pyplot as plt

# Generar 100 puntos espaciados para la curva suave
x_curve = np.linspace(min(x) - 1, max(x) + 1, 100)

# Evaluar el polinomio P(x)
# np.polyval requiere los coeficientes en orden DECRECIENTE (a3, a2, a1, a0)
a_rev = a[::-1] 
y_curve = np.polyval(a_rev, x_curve)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(x_curve, y_curve, label='Polinomio Interpolante P(x)', color='blue')
plt.plot(x, y, 'ro', label='Puntos de Interpolación') 
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polinomio de Interpolación (Método Vandermonde)')
plt.grid(True)
plt.legend()
plt.show()