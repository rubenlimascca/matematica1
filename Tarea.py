import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import psutil
import os

def generar_sistema_spd_disperso(n, densidad=0.001, semilla=None):
    """
    Genera un sistema lineal Ax=b con A dispersa, simétrica y definida positiva
    
    Parámetros:
    -----------
    n : int
        Número de ecuaciones (tamaño del sistema)
    densidad : float
        Densidad de la matriz (elementos no cero)
    semilla : int, opcional
        Semilla para reproducibilidad
    
    Retorna:
    --------
    A : sparse matrix CSR
        Matriz del sistema (SPD)
    b : ndarray
        Vector del lado derecho
    x_sol : ndarray
        Solución verdadera (para verificación)
    """
    if semilla is not None:
        np.random.seed(semilla)
    
    print(f"Generando sistema {n}x{n} con densidad {densidad}...")
    
    # Generar matriz dispersa aleatoria
    A = sp.random(n, n, density=densidad, format='csr', random_state=semilla)
    
    # Hacerla simétrica
    A = A + A.T
    
    # Asegurar que sea diagonalmente dominante y definida positiva
    diagonal = sp.diags(np.array(A.sum(axis=1)).flatten() + n)
    A = A + diagonal
    
    # Generar solución verdadera
    x_sol = np.random.randn(n)
    
    # Calcular vector b correspondiente
    b = A @ x_sol
    
    return A, b, x_sol

def resolver_sistema_disperso(A, b):
    """
    Resuelve el sistema Ax=b usando métodos para matrices dispersas
    
    Parámetros:
    -----------
    A : sparse matrix
        Matriz del sistema en formato CSR
    b : ndarray
        Vector del lado derecho
    
    Retorna:
    --------
    x : ndarray
        Solución del sistema
    tiempo : float
        Tiempo de ejecución
    memoria : float
        Memoria utilizada (MB)
    """
    print("Resolviendo con método disperso...")
    
    # Medir memoria inicial
    proceso = psutil.Process(os.getpid())
    mem_inicial = proceso.memory_info().rss / 1024 / 1024  # MB
    
    # Medir tiempo
    inicio = time.time()
    
    try:
        # Usar solver directo para matrices dispersas
        x = spla.spsolve(A, b)
    except Exception as e:
        print(f"Error con spsolve: {e}")
        print("Intentando con factorización LU...")
        # Alternativa: factorización LU
        lu = spla.splu(A.tocsc())
        x = lu.solve(b)
    
    tiempo = time.time() - inicio
    
    # Medir memoria final
    mem_final = proceso.memory_info().rss / 1024 / 1024
    memoria = mem_final - mem_inicial
    
    return x, tiempo, memoria

def resolver_sistema_denso(A, b):
    """
    Resuelve el sistema Ax=b convirtiendo a formato denso
    
    Parámetros:
    -----------
    A : sparse matrix
        Matriz del sistema
    b : ndarray
        Vector del lado derecho
    
    Retorna:
    --------
    x : ndarray
        Solución del sistema
    tiempo : float
        Tiempo de ejecución
    memoria : float
        Memoria utilizada (MB)
    """
    print("Convirtiendo a formato denso y resolviendo...")
    
    proceso = psutil.Process(os.getpid())
    mem_inicial = proceso.memory_info().rss / 1024 / 1024
    
    inicio = time.time()
    
    # Convertir a formato denso
    A_denso = A.toarray()
    b_denso = b.copy()
    
    # Resolver sistema denso
    x = la.solve(A_denso, b_denso, assume_a='sym')
    
    tiempo = time.time() - inicio
    
    mem_final = proceso.memory_info().rss / 1024 / 1024
    memoria = mem_final - mem_inicial
    
    return x, tiempo, memoria

def comparar_metodos(n=10000, densidad=0.001, semilla=42):
    """
    Compara los métodos disperso y denso para resolver el sistema lineal
    """
    print("=" * 60)
    print(f"COMPARACIÓN DE MÉTODOS PARA SISTEMA {n}x{n}")
    print("=" * 60)
    
    # Generar sistema
    A, b, x_verdadera = generar_sistema_spd_disperso(n, densidad, semilla)
    
    print(f"\nMatriz A:")
    print(f"  - Dimensiones: {A.shape}")
    print(f"  - Elementos no cero: {A.nnz}")
    print(f"  - Densidad: {A.nnz / (n*n):.6f}")
    print(f"  - Memoria estimada CSR: {A.data.nbytes + A.indices.nbytes + A.indptr.nbytes} bytes")
    print(f"  - Memoria estimada densa: {(n*n) * 8} bytes")
    
    # Resolver con método disperso
    x_disperso, t_disperso, mem_disperso = resolver_sistema_disperso(A, b)
    
    # Resolver con método denso
    x_denso, t_denso, mem_denso = resolver_sistema_denso(A, b)
    
    # Calcular errores
    error_disperso = np.linalg.norm(x_disperso - x_verdadera) / np.linalg.norm(x_verdadera)
    error_denso = np.linalg.norm(x_denso - x_verdadera) / np.linalg.norm(x_verdadera)
    
    error_relativo = np.linalg.norm(x_disperso - x_denso) / np.linalg.norm(x_denso)
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS:")
    print("=" * 60)
    
    print(f"\nPRECISIÓN:")
    print(f"  - Error relativo disperso: {error_disperso:.2e}")
    print(f"  - Error relativo denso: {error_denso:.2e}")
    print(f"  - Diferencia entre soluciones: {error_relativo:.2e}")
    
    print(f"\nTIEMPO DE EJECUCIÓN:")
    print(f"  - Método disperso: {t_disperso:.4f} segundos")
    print(f"  - Método denso: {t_denso:.4f} segundos")
    print(f"  - Speedup: {t_denso/t_disperso:.2f}x")
    
    print(f"\nMEMORIA UTILIZADA:")
    print(f"  - Método disperso: {mem_disperso:.2f} MB")
    print(f"  - Método denso: {mem_denso:.2f} MB")
    print(f"  - Ratio de memoria: {mem_denso/mem_disperso:.2f}x")
    
    # Verificar que la solución satisface Ax = b
    residuo_disperso = np.linalg.norm(A @ x_disperso - b) / np.linalg.norm(b)
    residuo_denso = np.linalg.norm(A @ x_denso - b) / np.linalg.norm(b)
    
    print(f"\nRESIDUOS (||Ax - b|| / ||b||):")
    print(f"  - Método disperso: {residuo_disperso:.2e}")
    print(f"  - Método denso: {residuo_denso:.2e}")
    
    return {
        'A': A, 'b': b, 'x_verdadera': x_verdadera,
        'x_disperso': x_disperso, 'x_denso': x_denso,
        'tiempos': {'disperso': t_disperso, 'denso': t_denso},
        'memoria': {'disperso': mem_disperso, 'denso': mem_denso},
        'errores': {'disperso': error_disperso, 'denso': error_denso}
    }

def visualizar_sistema(A, b, resultados, n_muestras=1000):
    """
    Visualiza aspectos del sistema y las soluciones
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Patrón de dispersión de A
    ax1.spy(A, markersize=0.1, alpha=0.6)
    ax1.set_title(f'Patrón de dispersión de A ({A.shape[0]}x{A.shape[1]})')
    ax1.set_xlabel('Columnas')
    ax1.set_ylabel('Filas')
    
    # 2. Comparación de soluciones (primeras n_muestras componentes)
    n_muestras = min(n_muestras, len(resultados['x_verdadera']))
    indices = np.arange(n_muestras)
    
    ax2.plot(indices, resultados['x_verdadera'][:n_muestras], 'k-', 
             linewidth=2, label='Solución verdadera', alpha=0.7)
    ax2.plot(indices, resultados['x_disperso'][:n_muestras], 'ro', 
             markersize=2, label='Solución dispersa', alpha=0.6)
    ax2.plot(indices, resultados['x_denso'][:n_muestras], 'b+', 
             markersize=3, label='Solución densa', alpha=0.6)
    ax2.set_title('Comparación de soluciones (primeras componentes)')
    ax2.set_xlabel('Componente')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Gráfico de barras comparativo
    metodos = ['Disperso', 'Denso']
    tiempos = [resultados['tiempos']['disperso'], resultados['tiempos']['denso']]
    memoria = [resultados['memoria']['disperso'], resultados['memoria']['denso']]
    
    x = np.arange(len(metodos))
    width = 0.35
    
    ax3.bar(x - width/2, tiempos, width, label='Tiempo (s)', alpha=0.7)
    ax3.bar(x + width/2, memoria, width, label='Memoria (MB)', alpha=0.7)
    ax3.set_xlabel('Método')
    ax3.set_ylabel('Tiempo (s) / Memoria (MB)')
    ax3.set_title('Comparación de rendimiento')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metodos)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución del vector b
    ax4.hist(b, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('Distribución del vector b')
    ax4.set_xlabel('Valor')
    ax4.set_ylabel('Frecuencia')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def prueba_diferentes_tamanos(tamanos=[1000, 5000, 10000], densidad=0.001):
    """
    Prueba el rendimiento con diferentes tamaños de sistema
    """
    resultados_comparativos = []
    
    for n in tamanos:
        print(f"\n{'#'*60}")
        print(f"PROBANDO CON n = {n}")
        print(f"{'#'*60}")
        
        try:
            resultado = comparar_metodos(n, densidad)
            resultados_comparativos.append({
                'n': n,
                'tiempo_disperso': resultado['tiempos']['disperso'],
                'tiempo_denso': resultado['tiempos']['denso'],
                'memoria_disperso': resultado['memoria']['disperso'],
                'memoria_denso': resultado['memoria']['denso']
            })
        except Exception as e:
            print(f"Error con n={n}: {e}")
            continue
    
    # Graficar resultados comparativos
    if resultados_comparativos:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        n_vals = [r['n'] for r in resultados_comparativos]
        tiempo_disperso = [r['tiempo_disperso'] for r in resultados_comparativos]
        tiempo_denso = [r['tiempo_denso'] for r in resultados_comparativos]
        memoria_disperso = [r['memoria_disperso'] for r in resultados_comparativos]
        memoria_denso = [r['memoria_denso'] for r in resultados_comparativos]
        
        # Tiempos
        ax1.plot(n_vals, tiempo_disperso, 'bo-', label='Método disperso', linewidth=2)
        ax1.plot(n_vals, tiempo_denso, 'ro-', label='Método denso', linewidth=2)
        ax1.set_xlabel('Tamaño del sistema (n)')
        ax1.set_ylabel('Tiempo (segundos)')
        ax1.set_title('Tiempo de ejecución vs Tamaño del sistema')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Memoria
        ax2.plot(n_vals, memoria_disperso, 'bo-', label='Método disperso', linewidth=2)
        ax2.plot(n_vals, memoria_denso, 'ro-', label='Método denso', linewidth=2)
        ax2.set_xlabel('Tamaño del sistema (n)')
        ax2.set_ylabel('Memoria (MB)')
        ax2.set_title('Memoria utilizada vs Tamaño del sistema')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()

# Ejecutar ejemplo principal
if __name__ == "__main__":
    # Ejemplo con n=10000
    print("EJECUTANDO SISTEMA CON n=10000")
    resultados = comparar_metodos(n=10000, densidad=0.001, semilla=42)
    
    # Visualizar resultados
    visualizar_sistema(resultados['A'], resultados['b'], resultados)
    
    # Prueba con diferentes tamaños (opcional, comentado por tiempo de ejecución)
    # print("\n\nPRUEBA CON DIFERENTES TAMAÑOS")
    # prueba_diferentes_tamanos(tamanos=[1000, 5000, 10000])
    
    # Ejemplo adicional más pequeño para visualización detallada
    print("\n\nEJEMPLO PEQUEÑO PARA VISUALIZACIÓN DETALLADA")
    resultados_pequeno = comparar_metodos(n=500, densidad=0.05, semilla=123)
    visualizar_sistema(resultados_pequeno['A'], resultados_pequeno['b'], resultados_pequeno)


    
# Para ejecutar solo el sistema de 10 ecuaciones
resultados = comparar_metodos(n=10, densidad=0.001)

# Para probar con diferentes tamaños
prueba_diferentes_tamanos([1000, 5000, 10000])