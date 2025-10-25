import numpy as np
import libreria as lib

n=4
A=np.random.rand(n,n)

LU=lib.LUdecomposition(A)

L=LU[0]
print("L:\n",L)
U=LU[1]
print("U:\n",U)
print("A-LU: \n",A-L@U)
