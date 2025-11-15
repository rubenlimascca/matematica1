import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generar_matriz_spd_aleatoria(n, densidad=0.1, semilla=None, metodo='chol'):
    """
    Genera una matriz dispersa simétrica definida positiva (SPD) aleatoria en formato CSR
    
    Parámetros:
    -----------
    n : int
        Dimensión de la matriz (n x n)
    densidad : float (0-1)
        Densidad de elementos no cero
    semilla : int, opcional
        Semilla para reproducibilidad
    metodo : str
        Método de generación: 'chol', 'diag_dom', 'eigen'
    
    Retorna:
    --------
    A : scipy.sparse matrix en formato CSR
        Matriz dispersa simétrica definida positiva
    """
    if semilla is not None:
        np.random.seed(semilla)
    
    if metodo == 'chol':
        # Método 1: Usando factorización de Cholesky
        # Generar matriz dispersa aleatoria
        G = sp.random(n, n, density=densidad, format='csr')
        
        # Hacerla simétrica: A = G + G.T
        A = G + G.T
        
        # Asegurar que sea definida positiva agregando múltiplo de la identidad
        # Calculamos el valor absoluto del eigenvalor más pequeño (aproximado)
        radio_espectral = norm(A, ord=2)  # Estimación del radio espectral
        diag_shift = radio_espectral + 0.1  # Shift para asegurar definida positiva
        
        # Agregar a la diagonal
        A = A + sp.eye(n) * diag_shift
        
    elif metodo == 'diag_dom':
        # Método 2: Matriz diagonalmente dominante
        A = sp.random(n, n, density=densidad, format='csr')
        
        # Hacer simétrica
        A = (A + A.T) * 0.5
        
        # Hacer diagonalmente dominante
        suma_filas = np.array(A.sum(axis=1)).flatten()
        for i in range(n):
            A[i, i] = suma_filas[i] + np.random.rand() + 1.0
            
    elif metodo == 'eigen':
        # Método 3: Usando descomposición espectral
        # Generar matriz dispersa aleatoria
        G = sp.random(n, n, density=densidad, format='csr')
        A = G + G.T  # Hacer simétrica
        
        # Calcular eigenvalores aproximados y ajustarlos para ser positivos
        try:
            # Usar eigs para matrices dispersas
            from scipy.sparse.linalg import eigs
            eigenvals, _ = eigs(A, k=min(6, n-1))
            min_eigval = np.min(np.real(eigenvals))
            
            if min_eigval <= 0:
                A = A + sp.eye(n) * (abs(min_eigval) + 0.1)
        except:
            # Fallback: método simple
            radio_espectral = norm(A, ord=2)
            A = A + sp.eye(n) * (radio_espectral + 0.1)
    
    else:
        raise ValueError("Método no reconocido. Usar 'chol', 'diag_dom' o 'eigen'")
    
    # Convertir a CSR y asegurar simetría
    A = A.tocsr()
    
    # Verificar que sea simétrica (aproximadamente)
    if not np.allclose(A.data, A.T.data, rtol=1e-10):
        A = (A + A.T) * 0.5
    
    return A

def verificar_spd(matriz):
    """
    Verifica si una matriz es simétrica y definida positiva
    
    Parámetros:
    -----------
    matriz : scipy.sparse matrix
    
    Retorna:
    --------
    es_spd : bool
        True si la matriz es SPD
    info : dict
        Información detallada de la verificación
    """
    info = {}
    
    # Verificar simetría
    diferencia = matriz - matriz.T
    info['simetria_error'] = norm(diferencia)
    es_simetrica = info['simetria_error'] < 1e-10
    
    # Verificar definida positiva (usando Cholesky)
    try:
        from scipy.sparse.linalg import splu
        # Intentar factorización Cholesky (LU para matrices simétricas)
        lu = splu(matriz.tocsc())
        info['cholesky_exitosa'] = True
        es_definida_positiva = True
    except:
        info['cholesky_exitosa'] = False
        es_definida_positiva = False
    
    # Verificar eigenvalores (para matrices pequeñas)
    if matriz.shape[0] <= 1000:
        try:
            from scipy.sparse.linalg import eigs
            eigenvals, _ = eigs(matriz, k=min(5, matriz.shape[0]-1))
            info['min_eigenvalor'] = np.min(np.real(eigenvals))
            info['max_eigenvalor'] = np.max(np.real(eigenvals))
            es_definida_positiva = es_definida_positiva and (info['min_eigenvalor'] > 0)
        except:
            info['eigen_calculo'] = 'Falló'
    
    info['es_spd'] = es_simetrica and es_definida_positiva
    return info['es_spd'], info

def visualizar_matriz_spd(matriz, titulo="Matriz SPD Dispersa", tamaño_figura=(12, 5)):
    """
    Visualiza una matriz dispersa SPD
    """
    matriz_densa = matriz.toarray()
    
    # Crear figura con subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=tamaño_figura)
    
    # Gráfico 1: Heatmap de la matriz
    cmap = ListedColormap(['white', 'blue'])
    im1 = ax1.imshow(matriz_densa != 0, cmap=cmap, aspect='auto')
    ax1.set_title(f'{titulo}\n(Patrón de dispersión)')
    ax1.set_xlabel('Columnas')
    ax1.set_ylabel('Filas')
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # Gráfico 2: Valores de la matriz (solo elementos no cero)
    non_zero_mask = matriz_densa != 0
    im2 = ax2.imshow(np.where(non_zero_mask, matriz_densa, np.nan), 
                     cmap='RdBu_r', aspect='auto')
    ax2.set_title('Valores de los elementos no cero')
    ax2.set_xlabel('Columnas')
    ax2.set_ylabel('Filas')
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    # Gráfico 3: Distribución de los valores
    valores_no_cero = matriz_densa[non_zero_mask]
    ax3.hist(valores_no_cero, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Distribución de valores no cero')
    ax3.set_xlabel('Valor')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Verificar propiedades SPD
    es_spd, info_spd = verificar_spd(matriz)
    
    # Mostrar información
    print(f"Información de la matriz SPD:")
    print(f"  Dimensiones: {matriz.shape[0]} x {matriz.shape[1]}")
    print(f"  Elementos totales: {matriz.shape[0] * matriz.shape[1]}")
    print(f"  Elementos no cero: {matriz.nnz}")
    print(f"  Densidad: {matriz.nnz / (matriz.shape[0] * matriz.shape[1]):.4f}")
    print(f"  ¿Es SPD?: {'SÍ' if es_spd else 'NO'}")
    
    if 'min_eigenvalor' in info_spd:
        print(f"  Eigenvalor mínimo: {info_spd['min_eigenvalor']:.6f}")
        print(f"  Eigenvalor máximo: {info_spd['max_eigenvalor']:.6f}")
    print(f"  Error de simetría: {info_spd['simetria_error']:.2e}")
    print(f"  Factorización Cholesky: {'Exitosa' if info_spd['cholesky_exitosa'] else 'Falló'}")

def comparar_matrices_spd(tamaños, densidades, semilla=42):
    """
    Genera y compara múltiples matrices SPD con diferentes parámetros
    """
    np.random.seed(semilla)
    
    fig, axes = plt.subplots(len(tamaños), len(densidades), 
                            figsize=(5*len(densidades), 4*len(tamaños)))
    
    if len(tamaños) == 1:
        axes = axes.reshape(1, -1)
    if len(densidades) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, n in enumerate(tamaños):
        for j, densidad in enumerate(densidades):
            # Generar matriz SPD
            matriz = generar_matriz_spd_aleatoria(n, densidad, semilla=semilla)
            matriz_densa = matriz.toarray()
            
            # Verificar SPD
            es_spd, info = verificar_spd(matriz)
            
            # Visualizar
            ax = axes[i, j]
            cmap = ListedColormap(['white', 'blue'])
            im = ax.imshow(matriz_densa != 0, cmap=cmap, aspect='auto')
            ax.set_title(f'{n}x{n}, densidad={densidad}\nSPD: {"SÍ" if es_spd else "NO"}')
            ax.set_xlabel('Columnas')
            ax.set_ylabel('Filas')
    
    plt.tight_layout()
    plt.show()

def benchmark_matrices_spd(tamaños, densidades, semilla=42):
    """
    Benchmark de matrices SPD generadas con diferentes métodos
    """
    metodos = ['chol', 'diag_dom', 'eigen']
    
    resultados = []
    for n in tamaños:
        for densidad in densidades:
            for metodo in metodos:
                try:
                    matriz = generar_matriz_spd_aleatoria(n, densidad, semilla=semilla, metodo=metodo)
                    es_spd, info = verificar_spd(matriz)
                    
                    resultados.append({
                        'tamaño': n,
                        'densidad': densidad,
                        'metodo': metodo,
                        'es_spd': es_spd,
                        'elementos_no_cero': matriz.nnz,
                        'densidad_real': matriz.nnz / (n * n),
                        'min_eigenvalor': info.get('min_eigenvalor', np.nan),
                        'error_simetria': info['simetria_error']
                    })
                except Exception as e:
                    print(f"Error con n={n}, densidad={densidad}, método={metodo}: {e}")
    
    return resultados

# Ejemplo de uso
if __name__ == "__main__":
    print("=== Ejemplo 1: Matriz SPD pequeña ===")
    matriz1 = generar_matriz_spd_aleatoria(1000, densidad=0.01, semilla=123, metodo='chol')
    visualizar_matriz_spd(matriz1, "Matriz SPD 1000x1000")
    
    print("\n=== Ejemplo 2: Matriz SPD más grande ===")
    matriz2 = generar_matriz_spd_aleatoria(50, densidad=0.1, semilla=456, metodo='diag_dom')
    visualizar_matriz_spd(matriz2, "Matriz SPD 50x50", tamaño_figura=(15, 5))
    
    print("\n=== Ejemplo 3: Comparación de matrices SPD ===")
    tamaños = [10, 20, 30]
    densidades = [0.2, 0.3, 0.4]
    comparar_matrices_spd(tamaños, densidades)
    
    print("\n=== Ejemplo 4: Benchmark de métodos ===")
    resultados = benchmark_matrices_spd([10, 20], [0.2, 0.3])
    for res in resultados:
        print(f"Tamaño: {res['tamaño']}, Densidad: {res['densidad']}, "
              f"Método: {res['metodo']}, SPD: {res['es_spd']}")
    
    print("\n=== Ejemplo 5: Propiedades CSR de matriz SPD ===")
    matriz_spd = generar_matriz_spd_aleatoria(8, densidad=0.4, semilla=789)
    print("Matriz SPD:")
    print(matriz_spd.toarray())
    print(f"\nEstructura CSR:")
    print(f"  data: {matriz_spd.data}")
    print(f"  indices: {matriz_spd.indices}")
    print(f"  indptr: {matriz_spd.indptr}")
    
    # Verificar simetría explícitamente
    print(f"\nVerificación de simetría:")
    print("¿A == A.T?:", np.allclose(matriz_spd.toarray(), matriz_spd.toarray().T))
    
    
    
    
    
    # Matriz SPD para problemas de elementos finitos
matriz_fem = generar_matriz_spd_aleatoria(100, densidad=0.05, metodo='diag_dom')

# Matriz SPD densa para pruebas numéricas
matriz_densa_spd = generar_matriz_spd_aleatoria(50, densidad=0.8, metodo='chol')

# Verificar condicionamiento
from scipy.sparse.linalg import eigs
eigenvals, _ = eigs(matriz_fem, k=2)
numero_condicion = abs(eigenvals[0]) / abs(eigenvals[1])
print(f"Número de condición: {numero_condicion:.2f}")