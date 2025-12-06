import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import rand, issparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm # Para la norma euclidiana

N = 1000 # Dimensión N x N
density = 0.01 # 1% de elementos no nulos

# 1. Generar una matriz dispersa A (inicialmente no simétrica)
A_sparse_temp = rand(N, N, density=density, format="csr")

# 2. Asegurar que sea simétrica (A = A + A^T)
# El resultado es una matriz dispersa
A_sparse = (A_sparse_temp + A_sparse_temp.T) 

# 3. Generar un vector b aleatorio
b = np.random.rand(N)

# 4. Visualizar el patrón de dispersión (Graficar A)
plt.figure(figsize=(5, 5))
plt.spy(A_sparse, markersize=1) # spy() muestra los elementos no nulos
plt.title('Patrón de Dispersión de la Matriz A (1000x1000)')
plt.show()

################################
# 1. Resolver el sistema disperso A * x_sparse = b
t0_sparse = time.time()
# spsolve es la función eficiente de SciPy para sistemas dispersos
x_sparse = spsolve(A_sparse, b) 
t_sparse = time.time() - t0_sparse

# 2. Calcular la norma euclidiana de la solución ||x||_2
norm_x_sparse = norm(x_sparse) 

print("--- Resolución Sistema Disperso ---")
print(f"Tiempo de resolución (Disperso): {t_sparse:.5f} segundos")
print(f"Norma Euclidea de la solución ||x||_2: {norm_x_sparse:.4f}")




#################################
# 1. Transformar A_sparse en A_dense (matriz densa/llena)
A_dense = A_sparse.toarray()

# 2. Resolver el sistema denso A_dense * x_dense = b
t0_dense = time.time()
# np.linalg.solve utiliza métodos optimizados para matrices densas
x_dense = np.linalg.solve(A_dense, b) 
t_dense = time.time() - t0_dense

# 3. Calcular la norma euclidiana del residuo ||r||_2
# r = b - A_dense * x_dense
# np.dot realiza la multiplicación matricial A*x
residue = b - np.dot(A_dense, x_dense)
norm_residue_dense = norm(residue)

print("\n--- Resolución Sistema Denso ---")
print(f"Tiempo de resolución (Denso): {t_dense:.5f} segundos")
print(f"Norma Euclidea del residuo ||r||_2: {norm_residue_dense:.2e}") 

print(f"\n--- Comparación ---")
# Comparación del tiempo de cálculo
if t_sparse < t_dense:
    print(f"La matriz Dispersa fue más rápida por un factor de: {t_dense/t_sparse:.2f} veces.")
else:
    print("La matriz Densa fue más rápida (esto puede ocurrir si la matriz es pequeña o si la densidad es alta).")