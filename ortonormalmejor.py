import numpy as np


def gram_schmidt_robusto(vectores, tol=1e-10):
    """
    Implementación robusta de Gram-Schmidt con manejo de dependencia lineal.

    Args:
        vectores: Lista de vectores o matriz NumPy
        tol: Tolerancia para considerar un vector como linealmente dependiente

    Returns:
        Base ortonormal y lista de índices de vectores linealmente independientes
    """
    if isinstance(vectores, list):
        matriz = np.array(vectores).T
    else:
        matriz = vectores.copy()

    n_vectores = matriz.shape[1]
    base_ortonormal = []
    indices_independientes = []

    for i in range(n_vectores):
        v = matriz[:, i].astype(float)

        # Restar proyecciones sobre la base actual
        for u in base_ortonormal:
            proj = np.dot(u, v) * u
            v = v - proj

        # Verificar si el vector resultante es significativo
        norma = np.linalg.norm(v)
        if norma > tol:
            base_ortonormal.append(v / norma)
            indices_independientes.append(i)
        else:
            print(f"Vector {i+1} es linealmente dependiente de los anteriores")

    return np.array(base_ortonormal).T, indices_independientes


# Ejemplo de uso
vectores = np.array(
    [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]  # Este es linealmente dependiente
)

base, indices = gram_schmidt_robusto(vectores)
print("Base ortonormal:")
print(base)
print(f"Vectores linealmente independientes: {[i+1 for i in indices]}")
