import numpy as np


def resolver_sistema_qr(A, b):
    """
    Resuelve el sistema lineal Ax = b usando descomposición QR

    Parámetros:
    A: matriz de coeficientes (n x n)
    b: vector de términos independientes (n,)

    Retorna:
    x: solución del sistema
    """
    # Verificar que A sea cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")

    # Verificar dimensiones compatibles
    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensiones incompatibles entre A y b")

    # Descomposición QR de A
    Q, R = np.linalg.qr(A)

    # Resolver R x = Q^T b
    # Q^T b
    Qt_b = Q.T @ b

    # Resolver sistema triangular superior R x = Qt_b
    x = np.linalg.solve(R, Qt_b)

    return x


# Ejemplo de uso
if __name__ == "__main__":
    # Sistema de ejemplo: 2x + y = 5
    #                     x + 3y = 6
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 6], dtype=float)

    x = resolver_sistema_qr(A, b)
    print("Solución del sistema QR Numpy:")
    print(f"x = {x}")
    print(f"Verificación: A @ x = {A @ x}")
    print(f"Vector b original: {b}")


#####################################


def gram_schmidt(A):
    """
    Implementación del proceso de Gram-Schmidt para descomposición QR
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def resolver_sistema_qr_manual(A, b):
    """
    Resuelve el sistema lineal usando descomposición QR manual
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")

    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensiones incompatibles entre A y b")

    # Descomposición QR manual
    Q, R = gram_schmidt(A)

    # Q^T b
    Qt_b = Q.T @ b

    # Resolver sistema triangular superior (sustitución hacia atrás)
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = Qt_b[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]

    return x


# Ejemplo de uso
if __name__ == "__main__":
    # Sistema de ejemplo: 0x + y = 5
    #                     x + 3y = 6
    A = np.array([[0, 1], [1, 3]], dtype=float)
    b = np.array([5, 6], dtype=float)

    x = resolver_sistema_qr_manual(A, b)
    print("Solución del sistema QR manual:")
    print(f"x = {x}")
    print(f"Verificación: A @ x = {A @ x}")
    print(f"Vector b original: {b}")


#######################
import numpy as np


def resolver_sistema_qr_robusto(A, b, tol=1e-10):
    """
    Resuelve el sistema lineal Ax = b con verificación de errores

    Parámetros:
    A: matriz de coeficientes
    b: vector de términos independientes
    tol: tolerancia para verificar singularidad

    Retorna:
    x: solución del sistema
    info: diccionario con información del proceso
    """
    # Verificaciones iniciales
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")

    if A.shape[0] != b.shape[0]:
        raise ValueError("Dimensiones incompatibles entre A y b")

    # Descomposición QR
    Q, R = np.linalg.qr(A)

    # Verificar si la matriz es singular
    diag_R = np.abs(np.diag(R))
    if np.min(diag_R) < tol:
        raise ValueError("La matriz A es singular o casi singular")

    # Resolver el sistema
    Qt_b = Q.T @ b
    x = np.linalg.solve(R, Qt_b)

    # Calcular residuo
    residuo = np.linalg.norm(A @ x - b)

    info = {
        "condicionamiento": np.linalg.cond(A),
        "residuo": residuo,
        "rango_efectivo": np.sum(diag_R > tol),
    }

    return x, info


# Ejemplo completo de uso
if __name__ == "__main__":
    # Sistema de ejemplo 3x3
    A = np.array([[4, -1, 1], [-1, 4, -2], [1, -2, 4]], dtype=float)

    b = np.array([12, -1, 5], dtype=float)

    print("Sistema de ecuaciones:")
    print("Matriz A:")
    print(A)
    print("Vector b:", b)
    print()

    # Resolver con método robusto
    try:
        x, info = resolver_sistema_qr_robusto(A, b)

        print("Solución encontrada Mejorado:")
        print(f"x = {x}")
        print()

        print("Información del proceso:")
        print(f"Residuo: {info['residuo']:.2e}")
        print(f"Número de condición: {info['condicionamiento']:.2f}")
        print(f"Rango efectivo: {info['rango_efectivo']}")
        print()

        print("Verificación:")
        print(f"A @ x = {A @ x}")
        print(f"Vector b original = {b}")
        print(f"Error: {np.linalg.norm(A @ x - b):.2e}")

    except ValueError as e:
        print(f"Error: {e}")


##########################
import numpy as np


def resolver_minimos_cuadrados_qr(A, b):
    """
    Resuelve el sistema sobredeterminado Ax ≈ b usando descomposición QR
    (Método de mínimos cuadrados)
    """
    m, n = A.shape

    # Descomposición QR
    Q, R = np.linalg.qr(A)

    # Q^T b
    Qt_b = Q.T @ b

    # Tomar solo las primeras n componentes (R es n x n)
    R1 = R[:n, :]
    Qt_b1 = Qt_b[:n]

    # Resolver R1 x = Qt_b1
    x = np.linalg.solve(R1, Qt_b1)

    return x


# Ejemplo de mínimos cuadrados
if __name__ == "__main__":
    # Sistema sobredeterminado: más ecuaciones que incógnitas
    A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    b = np.array([1, 3, 3, 5], dtype=float)

    print("Sistema sobredeterminado:")
    print("A =")
    print(A)
    print("b =", b)
    print()

    x = resolver_minimos_cuadrados_qr(A, b)
    print("Solución de mínimos cuadrados:")
    print(f"x = {x}")
    print()

    print("Ajuste:")
    print(f"A @ x = {A @ x}")
    print(f"Vector b original = {b}")
    print(f"Error cuadrático: {np.linalg.norm(A @ x - b)**2:.4f}")
