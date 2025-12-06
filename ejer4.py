import numpy as np
import matplotlib.pyplot as plt

# A. Definir el Polinomio Original p(x)
# Coeficientes de p(x) = 2 - x + 2x^2 + x^3 + 3x^4
# np.polyval requiere el orden DECRECIENTE: [a4, a3, a2, a1, a0]
p_orig_coeffs = np.array([3, 1, 2, -1, 2]) 
original_degree = len(p_orig_coeffs) - 1 # Grado 4

# B. Generar Datos "Reales" y Datos Ruidosos
# 1. Generar 50 puntos de muestreo en el intervalo [-1, 1]
x_lsm = np.linspace(-1, 1, 50)

# 2. Calcular los valores "reales" y_real = p(x) (Vectorizado)
y_real_lsm = np.polyval(p_orig_coeffs, x_lsm)

# 3. Generar Ruido Aleatorio (Distribución normal) y Añadirlo
# Usamos una semilla para asegurar la reproducibilidad
np.random.seed(42) 
# Ruido Gaussiano con media 0 y desviación estándar (sigma) de 0.5
noise = np.random.normal(0, 0.5, x_lsm.shape)
y_noisy_lsm = y_real_lsm + noise # Datos ruidosos o "experimentales"

# C. Ajuste por Mínimos Cuadrados (LSM)
# np.polyfit(x, y, grado) calcula los coeficientes del polinomio 
# de grado 'original_degree' que mejor se ajusta a los datos (x_lsm, y_noisy_lsm).
coeffs_ajustados = np.polyfit(x_lsm, y_noisy_lsm, deg=original_degree)

# Evaluar el polinomio ajustado para la gráfica (Vectorizado)
y_ajustada = np.polyval(coeffs_ajustados, x_lsm)

# Imprimir resultados
print("--- 4. Ajuste por Mínimos Cuadrados (LSM) ---")
print(f"Polinomio Original (Grado {original_degree}): {p_orig_coeffs.round(4)}")
print(f"Coeficientes Ajustados (LSM): {coeffs_ajustados.round(4)}")
print("Observación: Los coeficientes ajustados son una aproximación de los originales \nque minimiza el error cuadrático con respecto a los datos ruidosos.")

######################

# Graficar
plt.figure(figsize=(9, 6))
plt.plot(x_lsm, y_real_lsm, 'b--', label='Polinomio Original $p(x)$ (Sin Ruido)', linewidth=2)
plt.plot(x_lsm, y_noisy_lsm, 'ro', markersize=4, alpha=0.5, label='Datos Ruidosos')
plt.plot(x_lsm, y_ajustada, 'g-', label='Polinomio Ajustado por LSM', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simulación de Ajuste por Mínimos Cuadrados con Ruido')
plt.grid(True)
plt.legend()
plt.show()