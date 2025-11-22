import numpy as np
import numpy.polynomial as P

def intercambiaFilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:]=A[[fil_i,fil_j],:]
    
    
def operacionFila(A,fil_m,fil_piv,factor): #fil_m=fil_m-factor*fil_piv
    A[fil_m,:]=A[fil_m,:]-factor*A[fil_piv,:]
    
    def reescalaFila(A,fil_m,factor):
        A[fil_m,:]=factor*A[fil_m,:]
        
        
def escalonaSimple(A):
    nfil=A.shape[0]
    ncol=A.shape[1]
    
    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio=A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
            
            
def escalonaConPiv(A):
    nfil=A.shape[0]
    ncol=A.shape[1]
    for j in range(0,nfil):
        imax=np.argmax(np.abs(A[j:nfil,j]))
        intercambiaFilas(A,j+imax,j)
        for i in range(j+1,nfil):
            ratio=A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
            
def sustRegresiva(A,b):   #Resuelve un sistema escalonado
    N=b.shape[0]  # A y b deben ser array numpy bidiminsional
    x=np.zeros((N,1))
    for i in range(N-1,-1,-1):
        x[i,0]=(b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
    return x       # Array bidimensional


def sustProgresiva(A,b):   #Resuelve un sistema escalonado
    N=b.shape[0]  # A y b deben ser array numpy bidiminsional
    x=np.zeros((N,1))
    for i in range(0,N):
        x[i,0]=(b[i,0]-np.dot(A[i,0:i],x[0:i,0]))/A[i,i]
    return x # Array bidimensional
  
  
def GaussElimSimple(A,b):
    Ab=np.append(A,b,axis=1)  # matriz aumentada
    escalonaSimple(Ab)
    A1=Ab[:,0:Ab.shape[1]-1].copy()
    b1=Ab[:,Ab.shape[1]-1].copy()
    b1=b1.reshape(b.shape[0],1)
    x=sustRegresiva(A1,b1)
    return x # Array bidemensional

def GaussElimWithPiv(A,b):
    Ab=np.append(A,b,axis=1)  # matriz aumentada
    escalonaConPiv(Ab)
    A1=Ab[:,0:Ab.shape[1]-1].copy()
    b1=Ab[:,Ab.shape[1]-1].copy()
    b1=b1.reshape(b.shape[0],1)
    x=sustRegresiva(A1,b1)
    return x # Array bidemensional







# Creamos la matriz de hilbert
def hilbert_matrix(n):
    A=np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1]=1/(i+j-1)
    return A


def LUdecomposition(A):
    nrows=A.shape[0]  #numero de filas, estamos trabajando con una matriz cuadrada
    U=A.copy()
    L=np.eye(nrows,nrows,dtype=np.float64)
    
    for col in range(0,nrows-1):
        for row in range(col+1,nrows):
            mult=U[row,col]/U[col,col]
            L[row,col]=mult
            operacionFila(U,row,col, mult)

    return(L,U)



#def GaussElimPiv(A,b):
 #   Ab=np.append(A,b,axis=1)
  #  escalonaConPiv(Ab)
   # A1=Ab[:,0:Ab.shape[1]-1].copy

            

#adicionar 

def interpLagrange(cx,cy):
    n=len(cx)
    p=P.Polynomial([0])
    for i in range(n):
        mascara=np.ones(n,dtype=bool)
        mascara[i]=False
        raices=cx[mascara]
        Laux=P.Polynomial.fromroots(raices)
        p=p+cy[i]*Laux/Laux(cx[i])
    return p

#1

def SolveByLU(A,b):
    Y=sustProgresiva(L,b)
    X=sustRegresiva(U,Y)
    return X

def interpLagrange(cx,cy):
    n = len(cx)
    p = P.Polynomial(([0]))
    for i in range(n):
        mascara = np.ones(n,dtype=bool)
        mascara[i] = False
        raices = cx[mascara]
        Laux = P.Polynomial.fromroots(raices)
        p = p + cy[i]*Laux/Laux(cx[i])
    return p