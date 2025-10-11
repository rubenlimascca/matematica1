import numpy as np
import libreria as lib

#A=np.random.rand(4,5)
#print(A)



#lib.intercambiaFilas(A,0,3)


#lib.operacionFila(A,2,0,2)
#lib.escalonaSimple(A)

A=[[0, 14, -6], [12,0,4], [-11,3,0]]
lib.escalonaConPiv(A)
print(A)
