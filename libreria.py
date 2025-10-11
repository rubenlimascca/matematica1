import numpy as np

def intercambiaFilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:]=A[[fil_i,fil_j],:]
    
    
def operacionFila(A,fil_m,fil_piv,factor): #fil_m=fil_m-factor*fil_piv
    A[fil_m,:]=A[fil_m,:]-factor*A[fil_piv,:]
    
    def reescalaFila(A,fil_m,factor):
        A[fil_m,:]=factor*A[fil_m,:]