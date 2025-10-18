import numpy as np
import libreria as lib
import time

n=10000 # Tamaño del sistema lineal
A=np.random.rand(n,n)
b=np.random.rand(n,1)

start_time=time.perf_counter()
sol=lib.GaussElimSimple(A,b)
print(sol)
end_time=time.perf_counter()
elapsed_time=end_time-start_time
print(f"Tiempo transcurrido: {elapsed_time:.4f} segundos")

residuo=A@sol-b
print(residuo)
print("Norma del residuo:",np.linalg.norm(residuo))