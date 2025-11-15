import numpy as np


def gram_schmidt(vectores):
    """
    Realiza la ortonormalización de Gram-Schmidt sobre un conjunto de vectores.

    Args:
        vectores: Lista de vectores (arrays de NumPy) o matriz donde cada columna es un vector

    Returns:
        Lista de vectores ortonormales
    """
    # Convertir a array de NumPy si es necesario
    if isinstance(vectores, list):
        vectores = np.array(vectores).T  # Cada columna es un vector

    n = vectores.shape[1]  # Número de vectores
    ortogonales = np.zeros_like(vectores, dtype=float)

    for i in range(n):
        # Tomar el vector actual
        v = vectores[:, i].astype(float)

        # Restar las proyecciones sobre los vectores ya ortogonalizados
        for j in range(i):
            proj = np.dot(ortogonales[:, j], v) * ortogonales[:, j]
            v = v - proj

        # Normalizar el vector resultante
        norma = np.linalg.norm(v)
        if norma > 1e-10:  # Evitar división por cero
            ortogonales[:, i] = v / norma
        else:
            ortogonales[:, i] = v

    return ortogonales


# Ejemplo de uso
if __name__ == "__main__":
    # Vectores de ejemplo
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, 0, 1])
    v3 = np.array([0, 1, 1])

    vectores = [v1, v2, v3]
    base_ortonormal = gram_schmidt(vectores)

    print("Vectores originales:")
    for i, v in enumerate(vectores):
        print(f"v{i+1} = {v}")

    print("\nBase ortonormal:")
    for i, v in enumerate(base_ortonormal.T):
        print(f"u{i+1} = {v}")

    # Verificar ortogonalidad
    print("\nVerificación de ortogonalidad:")
    for i in range(len(vectores)):
        for j in range(i + 1, len(vectores)):
            producto = np.dot(base_ortonormal[:, i], base_ortonormal[:, j])
            print(f"u{i+1} · u{j+1} = {producto:.10f}")
