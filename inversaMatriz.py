import numpy as np

# Usando Usando NumPy


def inversa_matriz_numpy(matriz):
    """
    Calcula la inversa de una matriz usando NumPy

    Args:
        matriz: Array de NumPy o lista de listas

    Returns:
        Matriz inversa
    """
    matriz_np = np.array(matriz)

    # Verificar si la matriz es cuadrada
    if matriz_np.shape[0] != matriz_np.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")

    # Verificar si es invertible (determinante ≠ 0)
    if np.linalg.det(matriz_np) == 0:
        raise ValueError("La matriz no es invertible (determinante = 0)")

    return np.linalg.inv(matriz_np)


# Ejemplo de uso
matriz_ejemplo = [[1, 2], [3, 4]]
inversa = inversa_matriz_numpy(matriz_ejemplo)
print("Matriz original:")
print(np.array(matriz_ejemplo))
print("\nMatriz inversa usando NumPay:")
print(inversa)

###########################################################################


def inversa_gauss_jordan(matriz):
    """
    Calcula la inversa de una matriz usando eliminación de Gauss-Jordan

    Args:
        matriz: Lista de listas representando la matriz

    Returns:
        Matriz inversa
    """
    n = len(matriz)

    # Verificar que la matriz sea cuadrada
    if any(len(fila) != n for fila in matriz):
        raise ValueError("La matriz debe ser cuadrada")

    # Crear matriz aumentada [A|I]
    aumentada = []
    for i in range(n):
        fila = matriz[i][:]  # Copiar fila original
        fila.extend([1 if j == i else 0 for j in range(n)])  # Agregar identidad
        aumentada.append(fila)

    # Aplicar eliminación de Gauss-Jordan
    for i in range(n):
        # Pivoteo parcial
        max_row = max(range(i, n), key=lambda r: abs(aumentada[r][i]))
        if i != max_row:
            aumentada[i], aumentada[max_row] = aumentada[max_row], aumentada[i]

        # Verificar si el pivote es cero
        if abs(aumentada[i][i]) < 1e-10:
            raise ValueError("La matriz no es invertible")

        # Hacer el pivote igual a 1
        pivot = aumentada[i][i]
        for j in range(2 * n):
            aumentada[i][j] /= pivot

        # Eliminar otras filas
        for k in range(n):
            if k != i:
                factor = aumentada[k][i]
                for j in range(2 * n):
                    aumentada[k][j] -= factor * aumentada[i][j]

    # Extraer la inversa (parte derecha de la matriz aumentada)
    inversa = []
    for i in range(n):
        inversa.append(aumentada[i][n:])

    return inversa


# Ejemplo de uso
matriz_ejemplo = [[1, 2], [3, 4]]
inversa = inversa_gauss_jordan(matriz_ejemplo)
print("Matriz inversa (Gauss-Jordan):")
for fila in inversa:
    print([round(x, 6) for x in fila])


#################################################33


def inversa_matriz_completa(matriz, metodo="numpy"):
    """
    Función completa para calcular inversa con verificación

    Args:
        matriz: Matriz a invertir
        metodo: 'numpy' o 'gauss'

    Returns:
        Matriz inversa y información de verificación
    """
    try:
        if metodo == "numpy":
            inversa = inversa_matriz_numpy(matriz)
        else:
            inversa = inversa_gauss_jordan(matriz)

        # Verificar resultado: A * A⁻¹ ≈ I
        matriz_np = np.array(matriz)
        inversa_np = np.array(inversa)
        identidad_aproximada = np.dot(matriz_np, inversa_np)
        identidad_real = np.eye(len(matriz))

        error = np.max(np.abs(identidad_aproximada - identidad_real))

        print(f"Error de verificación: {error:.2e}")
        print("A * A⁻¹ ≈ I:", np.allclose(identidad_aproximada, identidad_real))

        return inversa

    except Exception as e:
        print(f"Error: {e}")
        return None


# Ejemplos de uso
print("=== Ejemplo 1: Matriz 2x2 ===")
matriz1 = [[4, 7], [2, 6]]
inversa1 = inversa_matriz_completa(matriz1, metodo="gauss")

print("\n=== Ejemplo 2: Matriz 3x3 ===")
matriz2 = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
inversa2 = inversa_matriz_completa(matriz2, metodo="numpy")


###########################


def invertir(matriz):
    """
    Función simplificada para invertir matrices
    """
    return np.linalg.inv(np.array(matriz))


# Uso rápido
A = [[2, 1], [1, 1]]
A_inv = invertir(A)
print("Matriz A:")
print(np.array(A))
print("Matriz inversa A⁻¹:")
print(A_inv)
print("Verificación A * A⁻¹:")
print(np.dot(np.array(A), A_inv))
