import numpy as np
import libreria as lib
import time

print("{:15s}{:25s}{:20s}".format(" n","cond","error"))
print("-"*50)

solutions=[]

for i in range(4,17):
    x=np.ones(i)
    H=lib.hilbert_matrix(i)
    b=H.dot(x)
    
    c=np.linalg.cond(H,2)
    xx=np.linalg.solve(H,b)
    err=np.linalg.norm(x-xx,np.inf)/np.linalg.norm(x,np.inf)
    solutions.append(xx)
    
    print("{:2d}{:20e}{:20e}".format(i,c,err))