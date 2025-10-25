import numpy as np
import libreria as lib
import time

n=1000
A=np.random.rand(n,n)
b=np.random.rand(n,1)

#X=lib.GaussElimWithPiv(A,b)
#print("AX-b:\n",A@X-b)
#print("||AX-b||_1:\n",np.linalg.norm(A@X-b,1))

start=time.perf_counter()
X=lib.GaussElimSimple(A,b)
end=time.perf_counter()
elapsed1=end-start
normErr1=np.linalg.norm(A@X-b,1)

start=time.perf_counter()
X=lib.GaussElimWithPiv(A,b)
end=time.perf_counter()
elapsed2=end-start
normErr2=np.linalg.norm(A@X-b,1)

start=time.perf_counter()
X=np.linalg.solve(A,b)
end=time.perf_counter()
elapsed3=end-start
normErr3=np.linalg.norm(A@X-b,1)

print("{:25s}{:25s}{:25s}".format("Error Gauss simple ","Error Gauss piv","Error linalg.solve"))
print("-"*70)
print("{:20e}{:20e}{:20e}".format(normErr1,normErr2,normErr3))


print("{:25s}{:25s}{:25s}".format("Time Gauss simple ","Time Gauss piv","Time linalg.solve"))
print("-"*70)
print("{:20e}{:20e}{:20e}".format(elapsed1,elapsed2,elapsed3))