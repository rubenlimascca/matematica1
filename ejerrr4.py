import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
# Se asume que 'libreria.py' contiene la función GaussElimWithPiv
# Esta línea fallará si el archivo 'libreria.py' no existe.
# Si no tienes la librería, podrías usar np.linalg.solve en su lugar 
# para simular la resolución del sistema lineal, pero mantendremos la estructura.
import libreria as bib 

# --- A. Definir el Polinomio Original y Generar Datos ---
# Polinomio base: p(x) = -2 + x +-x^2 + x^3 + 3x^4
a0, a1, a2, a3, a4 = -2, 1, -1, 1, 3
degree = 4

# Intervalo sobre el cual se efectúa el experimento
x = np.linspace(-1, 1, 100) 
# Valores exactos (sin ruido) del polinomio original
y_exact = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4

# Simular datos ruidosos
m = 100 # Número de puntos de muestreo
# Genera m números aleatorios X entre -1 y 1
X = 1 - 2 * np.random.rand(m) 
# Generar Y ruidoso: Y = P(X) + Ruido
# 10 * np.random.randn(m) -> Ruido Gaussiano con desviación estándar de 10
Y = a0 + a1 * X + a2 * X**2 + a3 * X**3 + a4 * X**4 + 10 * np.random.randn(m) 

# --- B. Ajuste por Mínimos Cuadrados (Ecuación Normal) ---

# 1. Construir la Matriz de Diseño A (Matriz de Vandermonde truncada)
# La matriz A tiene la forma [X^0, X^1, X^2, X^3, X^4]. 
# Esta es una matriz de m x (degree + 1) -> 100 x 5
A = np.vstack([X**i for i in range(degree + 1)]).T 

# 2. Calcular la matriz del sistema de la Ecuación Normal: M = A^T * A
# M es una matriz de (degree + 1) x (degree + 1) -> 5 x 5
M = np.matmul(A.T, A)

# 3. Calcular el vector del lado derecho: b_ls = A^T * Y
# b_ls es un vector de (degree + 1) -> 5x1
b_ls = np.matmul(A.T, Y)

# 4. Resolver el sistema M * sol = b_ls usando Gauss con Pivoteo
# Nota: La función bib.GaussElimWithPiv debe devolver el vector solución (sol).
# Para usar esta función, a menudo se requiere que b_ls sea una columna (m x 1)
try:
    # Aseguramos la forma de columna si la función lo requiere
    b_ls_col = b_ls.reshape(-1, 1) 
    sol = bib.GaussElimWithPiv(M, b_ls_col)
    
    # sol es el vector de coeficientes ajustados: [a0_fit, a1_fit, ..., a4_fit]
    # Si sol viene anidado (ej. [[a0], [a1], ...]), lo aplanamos.
    sol = np.array(sol).flatten() 
except Exception as e:
    print(f"\nADVERTENCIA: Fallo al usar bib.GaussElimWithPiv. Usando np.linalg.solve como alternativa.")
    print(f"Error específico: {e}")
    # Usar np.linalg.solve si la librería custom falla, respetando la matriz M.
    sol = np.linalg.solve(M, b_ls)

# --- C. Graficar Resultados ---

# Generar la curva del polinomio ajustado
# Coeficientes: sol[0] = a0_fit, sol[1] = a1_fit, ...
y_fit = sol[0] + sol[1] * x + sol[2] * x**2 + sol[3] * x**3 + sol[4] * x**4

print("\n--- Resultados del Ajuste ---")
print(f"Coeficientes Reales (a0 a4): {np.array([a0, a1, a2, a3, a4])}")
print(f"Coeficientes Ajustados (LSM): {sol.round(4)}")


fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(X, Y, "go", alpha=0.5, label="Datos Simulados con Ruido") 
ax.plot(x, y_exact, "r", lw=3, label=f"Valor Real $P(x)$ (Grado {degree})") 
ax.plot(x, y_fit, "b--", lw=2, label="Ajuste por Mínimos Cuadrados") 
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)
ax.set_title('Ajuste Polinómico por Mínimos Cuadrados (Ecuación Normal)')
ax.legend(loc='best')
plt.grid(True)
plt.show()