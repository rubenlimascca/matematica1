import numpy as np


def determinante_operaciones_elementales(matriz):
    """
    Calcula el determinante de una matriz usando operaciones elementales.

    Args:
        matriz: Lista de listas o array de numpy representando una matriz cuadrada

    Returns:
        float: El determinante de la matriz
    """
    # Convertir a numpy array si es necesario
    A = np.array(matriz, dtype=float)
    n = A.shape[0]

    # Verificar que la matriz sea cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")

    # Inicializar el determinante y el contador de intercambios
    det = 1.0
    intercambios = 0

    # Crear una copia para no modificar la matriz original
    M = A.copy()

    for i in range(n):
        # Pivoteo parcial: encontrar la fila con el máximo elemento en la columna i
        max_fila = i
        for k in range(i + 1, n):
            if abs(M[k, i]) > abs(M[max_fila, i]):
                max_fila = k

        # Si el pivote es cero, el determinante es cero
        if abs(M[max_fila, i]) < 1e-12:
            return 0.0

        # Intercambiar filas si es necesario
        if max_fila != i:
            M[[i, max_fila]] = M[[max_fila, i]]
            intercambios += 1
            det *= -1  # Cada intercambio cambia el signo del determinante

        # Hacer ceros debajo del pivote
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]

    # El determinante es el producto de los elementos de la diagonal
    for i in range(n):
        det *= M[i, i]

    return det


# Versión más detallada que muestra las operaciones paso a paso
def determinante_con_pasos(matriz):
    """
    Calcula el determinante mostrando las operaciones elementales paso a paso.
    """
    A = np.array(matriz, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")

    print("Matriz original:")
    print(A)
    print()

    det = 1.0
    intercambios = 0
    M = A.copy()

    for i in range(n):
        print(f"--- Paso {i+1} ---")
        print(f"Trabajando con la columna {i+1}")

        # Pivoteo parcial
        max_fila = i
        for k in range(i + 1, n):
            if abs(M[k, i]) > abs(M[max_fila, i]):
                max_fila = k

        print(f"Pivote máximo encontrado en fila {max_fila + 1}: {M[max_fila, i]}")

        if abs(M[max_fila, i]) < 1e-12:
            print("Pivote cero encontrado. Determinante = 0")
            return 0.0

        # Intercambiar filas si es necesario
        if max_fila != i:
            print(f"Intercambiando fila {i+1} con fila {max_fila+1}")
            M[[i, max_fila]] = M[[max_fila, i]]
            intercambios += 1
            det *= -1
            print("Matriz después del intercambio:")
            print(M)

        # Eliminación gaussiana
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            print(f"Eliminando elemento ({j+1},{i+1}) con factor: {factor:.4f}")
            M[j, i:] -= factor * M[i, i:]

        print("Matriz después de la eliminación:")
        print(M)
        print()

    # Calcular determinante final
    print("Matriz triangular final:")
    print(M)

    for i in range(n):
        det *= M[i, i]

    print(
        f"\nProducto de la diagonal: {' × '.join([f'{M[i,i]:.4f}' for i in range(n)])}"
    )
    print(f"Número de intercambios: {intercambios}")
    print(f"Determinante final: {det:.6f}")

    return det


# Función para verificar con numpy
def verificar_determinante(matriz):
    """Verifica el resultado usando numpy.linalg.det"""
    A = np.array(matriz, dtype=float)
    det_numpy = np.linalg.det(A)
    det_custom = determinante_operaciones_elementales(matriz)

    print(f"Determinante (numpy): {det_numpy:.6f}")
    print(f"Determinante (custom): {det_custom:.6f}")
    print(f"¿Coinciden?: {np.isclose(det_numpy, det_custom)}")

    return det_numpy, det_custom


##################################

# Ejemplo 1: Matriz 2x2
print("=== Ejemplo 1: Matriz 2x2 ===")
matriz_2x2 = [[2, 1], [1, 3]]
determinante_con_pasos(matriz_2x2)
print()

# Ejemplo 2: Matriz 3x3
print("=== Ejemplo 2: Matriz 3x3 ===")
matriz_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
det = determinante_operaciones_elementales(matriz_3x3)
print(f"Determinante: {det}")
print()

# Ejemplo 3: Matriz con intercambios
print("=== Ejemplo 3: Matriz que requiere intercambios ===")
matriz_intercambio = [[0, 2, 1], [1, 3, 4], [2, 1, 1]]
determinante_con_pasos(matriz_intercambio)
print()

# Ejemplo 4: Verificación con numpy
print("=== Ejemplo 4: Verificación ===")
matriz_verificar = [[4, 1, 2], [3, 2, 1], [1, 5, 3]]
verificar_determinante(matriz_verificar)


#################################################


# Manejo de casos especiales
def determinante_robusto(matriz):
    """
    Versión más robusta que maneja casos especiales.
    """
    try:
        A = np.array(matriz, dtype=float)

        # Matriz vacía
        if A.size == 0:
            return 1.0

        # Matriz 1x1
        if A.shape == (1, 1):
            return A[0, 0]

        # Verificar si es cuadrada
        if A.shape[0] != A.shape[1]:
            raise ValueError("La matriz debe ser cuadrada")

        return determinante_operaciones_elementales(A)

    except Exception as e:
        print(f"Error calculando el determinante: {e}")
        return None


# Ejemplo de uso robusto
matriz_1x1 = [[5]]
print(f"Determinante 1x1: {determinante_robusto(matriz_1x1)}")

matriz_vacia = []
print(f"Determinante matriz vacía: {determinante_robusto(matriz_vacia)}")
