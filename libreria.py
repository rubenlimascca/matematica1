import numpy as np

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

