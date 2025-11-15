import numpy as np
import ortonormalizacion


def gram_schmidt_completo(vectores):
    """
    Gram-Schmidt con verificación de las propiedades ortonormales.
    """
    base = ortonormalizacion.gram_schmidt(vectores)

    print("Propiedades de la base ortonormal:")

    # Verificar ortogonalidad
    n = base.shape[1]
    for i in range(n):
        for j in range(i, n):
            producto = np.dot(base[:, i], base[:, j])
            if i == j:
                print(f"||u{i+1}|| = {np.sqrt(producto):.6f} (debería ser 1)")
            else:
                print(f"u{i+1} · u{j+1} = {producto:.10f} (debería ser 0)")

    return base


# Ejemplo con vectores en R³
v1 = np.array([3, 1, 1])
v2 = np.array([1, 1, 0])
v3 = np.array([2, 0, 1])

base = gram_schmidt_completo([v1, v2, v3])
