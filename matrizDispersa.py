import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def generar_matriz_dispersa_aleatoria(filas, columnas, densidad=0.1, formato='csr', semilla=None):
    """
    Genera una matriz dispersa aleatoria en formato CSR
    
    Parámetros:
    -----------
    filas : int
        Número de filas de la matriz
    columnas : int
        Número de columnas de la matriz
    densidad : float (0-1)
        Densidad de elementos no cero (por defecto 0.1 = 10%)
    formato : str
        Formato de la matriz dispersa ('csr', 'csc', 'coo', etc.)
    semilla : int, opcional
        Semilla para reproducibilidad
    
    Retorna:
    --------
    matriz : scipy.sparse matrix
        Matriz dispersa en el formato especificado
    """
    if semilla is not None:
        np.random.seed(semilla)
    
    # Generar matriz dispersa aleatoria
    matriz = sp.random(filas, columnas, density=densidad, format=formato)
    
    # Convertir a CSR si no está en ese formato
    if formato != 'csr':
        matriz = matriz.tocsr()
    
    return matriz

def visualizar_matriz_dispersa(matriz, titulo="Matriz Dispersa", tamaño_figura=(10, 8)):
    """
    Visualiza una matriz dispersa
    
    Parámetros:
    -----------
    matriz : scipy.sparse matrix
        Matriz dispersa a visualizar
    titulo : str
        Título del gráfico
    tamaño_figura : tuple
        Tamaño de la figura (ancho, alto)
    """
    # Convertir a array denso para visualización
    matriz_densa = matriz.toarray()
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tamaño_figura)
    
    # Gráfico 1: Heatmap de la matriz
    cmap = ListedColormap(['white', 'blue'])
    im1 = ax1.imshow(matriz_densa != 0, cmap=cmap, aspect='auto')
    ax1.set_title(f'{titulo}\n(Elementos no cero)')
    ax1.set_xlabel('Columnas')
    ax1.set_ylabel('Filas')
    
    # Agregar barra de color para el primer gráfico
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # Gráfico 2: Patrón de dispersión
    filas, cols = matriz.nonzero()
    ax2.scatter(cols, filas, color='red', s=1, alpha=0.6)
    ax2.set_title('Patrón de Dispersión')
    ax2.set_xlabel('Columnas')
    ax2.set_ylabel('Filas')
    ax2.set_xlim(0, matriz.shape[1])
    ax2.set_ylim(0, matriz.shape[0])
    ax2.invert_yaxis()  # Para que coincida con la orientación de imshow
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar información de la matriz
    print(f"Información de la matriz:")
    print(f"  Dimensiones: {matriz.shape[0]} x {matriz.shape[1]}")
    print(f"  Elementos totales: {matriz.shape[0] * matriz.shape[1]}")
    print(f"  Elementos no cero: {matriz.nnz}")
    print(f"  Densidad: {matriz.nnz / (matriz.shape[0] * matriz.shape[1]):.4f}")
    print(f"  Formato: {matriz.format}")

def comparar_matrices_dispersas(tamaños, densidades, semilla=42):
    """
    Genera y compara múltiples matrices dispersas con diferentes parámetros
    
    Parámetros:
    -----------
    tamaños : list of tuples
        Lista de tuplas (filas, columnas)
    densidades : list of floats
        Lista de densidades a probar
    semilla : int
        Semilla para reproducibilidad
    """
    np.random.seed(semilla)
    
    n_ejemplos = len(tamaños) * len(densidades)
    fig, axes = plt.subplots(len(tamaños), len(densidades), 
                            figsize=(4*len(densidades), 4*len(tamaños)))
    
    if len(tamaños) == 1:
        axes = axes.reshape(1, -1)
    if len(densidades) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (filas, columnas) in enumerate(tamaños):
        for j, densidad in enumerate(densidades):
            # Generar matriz
            matriz = generar_matriz_dispersa_aleatoria(filas, columnas, densidad, semilla=semilla)
            matriz_densa = matriz.toarray()
            
            # Visualizar
            ax = axes[i, j]
            cmap = ListedColormap(['white', 'blue'])
            ax.imshow(matriz_densa != 0, cmap=cmap, aspect='auto')
            ax.set_title(f'{filas}x{columnas}, densidad={densidad}\n({matriz.nnz} elementos no cero)')
            ax.set_xlabel('Columnas')
            ax.set_ylabel('Filas')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo 1: Matriz individual
    print("=== Ejemplo 1: Matriz dispersa individual ===")
    matriz1 = generar_matriz_dispersa_aleatoria(1000, 1000, densidad=0.001, semilla=123)
    visualizar_matriz_dispersa(matriz1, "Matriz Dispersa 20x15")
    
    # Ejemplo 2: Matriz más grande
    print("\n=== Ejemplo 2: Matriz más grande ===")
    matriz2 = generar_matriz_dispersa_aleatoria(50, 40, densidad=0.1, semilla=456)
    visualizar_matriz_dispersa(matriz2, "Matriz Dispersa 50x40", tamaño_figura=(12, 6))
    
    # Ejemplo 3: Comparación de múltiples matrices
    print("\n=== Ejemplo 3: Comparación de matrices ===")
    tamaños = [(10, 10), (20, 15), (30, 25)]
    densidades = [0.1, 0.2, 0.3]
    comparar_matrices_dispersas(tamaños, densidades)
    
    # Ejemplo 4: Trabajando con la matriz CSR
    print("\n=== Ejemplo 4: Propiedades CSR ===")
    matriz_csr = generar_matriz_dispersa_aleatoria(8, 6, densidad=0.3, semilla=789)
    print("Matriz CSR:")
    print(matriz_csr.toarray())
    print(f"\nDatos CSR:")
    print(f"  data: {matriz_csr.data}")
    print(f"  indices: {matriz_csr.indices}")
    print(f"  indptr: {matriz_csr.indptr}")