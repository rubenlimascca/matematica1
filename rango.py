import numpy as np
import libreria as lib


# Usando NumPy
def rango_matriz(matriz):
    """
    Calcula el rango de una matriz usando NumPy

    Args:
        matriz: Lista de listas o array de NumPy

    Returns:
        int: Rango de la matriz
    """
    matriz_np = np.array(matriz)
    return np.linalg.matrix_rank(matriz_np)


# Ejemplo de uso
matriz_ejemplo = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Ejemplo de uso
matriz = [[1, 2, 3], [0, 5, 6], [0, 0, 9]]

# Ejemplo de uso
matriz1 = [[1, 2, 3], [2, 4, 6], [0, 0, 0]]

print(f"Rango de la matriz: {rango_matriz(matriz_ejemplo)}")
print(f"Rango de la matriz: {rango_matriz(matriz)}")
print(f"Rango de la matriz: {rango_matriz(matriz1)}")


# Implementación manual usando eliminación gaussiana
def rango_matriz_manual(matriz):
    """
    Calcula el rango de una matriz usando eliminación gaussiana

    Args:
        matriz: Lista de listas representando la matriz

    Returns:
        int: Rango de la matriz
    """
    # Convertir a float para evitar problemas de división
    matriz = [[float(elemento) for elemento in fila] for fila in matriz]

    filas = len(matriz)
    if filas == 0:
        return 0

    columnas = len(matriz[0])
    rango = 0

    # Crear una copia para no modificar la original
    temp_matriz = [fila[:] for fila in matriz]

    for col in range(columnas):
        # Encontrar fila pivote
        fila_pivote = rango
        while fila_pivote < filas and abs(temp_matriz[fila_pivote][col]) < 1e-10:
            fila_pivote += 1

        if fila_pivote == filas:
            continue  # Columna sin pivote

        # Intercambiar filas si es necesario
        if fila_pivote != rango:
            temp_matriz[rango], temp_matriz[fila_pivote] = (
                temp_matriz[fila_pivote],
                temp_matriz[rango],
            )

        # Hacer ceros debajo del pivote
        for i in range(rango + 1, filas):
            factor = temp_matriz[i][col] / temp_matriz[rango][col]
            for j in range(col, columnas):
                temp_matriz[i][j] -= factor * temp_matriz[rango][j]

        rango += 1
        if rango == min(filas, columnas):
            break

    return rango


print(f"Rango de la matriz manual: {rango_matriz_manual(matriz_ejemplo)}")
print(f"Rango de la matriz manual: {rango_matriz_manual(matriz)}")
print(f"Rango de la matriz manual: {rango_matriz_manual(matriz1)}")


# Rango de la matriz con nuestra libreria


def rango_matriz_manual_lib(matriz):
    lib.escalonaSimple(matriz)
    return np.any(matriz, axis=1).sum()


# Ejemplo de uso
matriz_ejemplo = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Ejemplo de uso
matriz = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])

# Ejemplo de uso
matriz1 = np.array([[1, 2, 3], [2, 4, 6], [0, 0, 0]])

print(f"Rango de la matriz manual librería: {rango_matriz_manual_lib(matriz_ejemplo)}")
print(f"Rango de la matriz manual librería: {rango_matriz_manual_lib(matriz)}")
print(f"Rango de la matriz manual librería: {rango_matriz_manual_lib(matriz1)}")
