import numpy as np
import libreria as lib

# A=np.random.rand(4,5)
# print(A)


# lib.intercambiaFilas(A,0,3)


# lib.operacionFila(A,2,0,2)
# lib.escalonaSimple(A)

# A=np.array([[0, 14, -6], [12,0,4], [-11,3,0]])
# print(A)
# lib.escalonaConPiv(A)
# print(A)


# no esta bien verificar luego
A = np.array([[2, 14, -6], [12, 0, 4], [-11, 3, 0]])
b = np.array([[1], [2], [3]])

print(A)
print(b)

sol = lib.GaussElimSimple(A, b)
print("La solución es:", sol)

residuo = A @ sol - b
print(residuo)
print("Norma del residuo:", np.linalg.norm(residuo))


matriz1 = np.array([[1, 2, 3], [2, 4, 6], [0, 0, 0]])

lib.escalonaSimple(matriz1)
print("La matriz escalonada simple es:", matriz1)

lib.intercambiaFilas(matriz1, 0, 2)
print("Intercambio de fila:", matriz1)
