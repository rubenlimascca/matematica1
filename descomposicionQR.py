import numpy as np


def gram_schmidt_matriz(A):
    """
    Aplica Gram-Schmidt a las columnas de una matriz A.

    Args:
        A: Matriz donde cada columna es un vector

    Returns:
        Q: Matriz con columnas ortonormales
        R: Matriz triangular superior tal que A = QR
    """
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v

    return Q, R


# Ejemplo de factorización QR
A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)

Q, R = gram_schmidt_matriz(A)
print("Matriz original A:")
print(A)
print("\nMatriz Q (ortonormal):")
print(Q)
print("\nMatriz R (triangular superior):")
print(R)
print("\nVerificación A = QR:")
print(Q @ R)
