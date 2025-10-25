import numpy as np
import libreria as lib

n=4
A=np.random.rand(n,n)
b=np.random.rand(n,1)

X=lib.SolveByLU(A,b)


print("AX-b:\n",A@X-b)